import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from vllm_kunlun.models.qwen3_vl import (
    Qwen3_VisionMLP,
    Qwen3_VisionPatchMerger,
    Qwen3_VisionPatchEmbed,
    Qwen3_VisionTransformer,
    Qwen3_VisionBlock,
    Qwen3VLForConditionalGeneration,
    Qwen3LLMForCausalLM
)
target_get_pp_group = "vllm_kunlun.models.qwen3_vl.get_pp_group"
target_merge_multimodal = "vllm_kunlun.models.qwen3_vl.merge_multimodal_embeddings"

from vllm.config import VllmConfig, ModelConfig

# 定义一个继承自 nn.Module 的 Mock 类
class MockBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, **kwargs):
        return x

class TestQwen3VisionComponents:
    @pytest.fixture
    def mock_linear(self):
        with patch("vllm.distributed.parallel_state.get_tensor_model_parallel_rank", return_value=0), \
             patch("vllm.distributed.parallel_state.get_tp_group", return_value=MagicMock()), \
             patch("vllm.distributed.parallel_state.get_tensor_model_parallel_world_size", return_value=1):
            yield

    def test_mlp_forward(self, mock_linear):
        with patch.object(Qwen3_VisionMLP, "__init__", return_value=None):
            mlp = Qwen3_VisionMLP(in_features=128, hidden_features=512)
            
            fc1_mock = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], x.shape[1], 512))
            fc2_mock = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], x.shape[1], 128))
            act_mock = lambda x: x 

            mlp.linear_fc1 = fc1_mock
            mlp.linear_fc2 = fc2_mock
            mlp.act_fn = act_mock

            x = torch.randn(1, 10, 128)
            output = mlp.forward(x)

            assert output.shape == (1, 10, 128)
            assert fc1_mock.call_count == 1

    def test_patch_merger_forward(self, mock_linear):
        with patch.object(Qwen3_VisionPatchMerger, "__init__", return_value=None):
            merger = Qwen3_VisionPatchMerger(d_model=128, context_dim=64, spatial_merge_size=2)
            merger.use_postshuffle_norm = False
            merger.hidden_size = 64 * (2**2) # 256

            merger.norm = MagicMock(side_effect=lambda x: x)
            fc1_mock = MagicMock(side_effect=lambda x: (x, None))
            merger.linear_fc1 = fc1_mock
            fc2_mock = MagicMock(side_effect=lambda x: (torch.randn(x.shape[0], 128), None))
            merger.linear_fc2 = fc2_mock
            merger.act_fn = lambda x: x

            x = torch.randn(12, 64) 
            output = merger.forward(x)

            assert fc1_mock.called
            assert output.shape[-1] == 128


class TestQwen3VisionTransformer:
    @pytest.fixture
    def mock_vision_config(self):
        conf = MagicMock()
        conf.hidden_size = 128
        conf.num_heads = 4
        conf.num_position_embeddings = 1024
        conf.patch_size = 14
        conf.spatial_merge_size = 2
        conf.temporal_patch_size = 2
        conf.in_channels = 3
        conf.out_hidden_size = 256
        conf.intermediate_size = 512
        conf.hidden_act = "silu"
        conf.depth = 2
        conf.deepstack_visual_indexes = [0, 1] 
        return conf

    def test_vit_init_and_forward(self, mock_vision_config):
        with patch.object(Qwen3_VisionPatchEmbed, "__init__", return_value=None), \
             patch("torch.nn.Embedding"), \
             patch("vllm_kunlun.models.qwen3_vl.Qwen2_5_VisionRotaryEmbedding"), \
             patch.object(Qwen3_VisionPatchMerger, "__init__", return_value=None), \
             patch("vllm_kunlun.models.qwen3_vl.get_vit_attn_backend"), \
             patch("vllm_kunlun.models.qwen3_vl.check_upstream_fa_availability", return_value=True), \
             patch("vllm_kunlun.models.qwen3_vl.Qwen3_VisionBlock", side_effect=MockBlock):

            vit = Qwen3_VisionTransformer(vision_config=mock_vision_config)

            # [关键修复1] vit是真实Module，其子模块必须也是Module类型
            vit.patch_embed = MockBlock()
            # 挂载属性
            vit.patch_embed.proj = MagicMock()
            vit.patch_embed.proj.weight.device = torch.device("cpu")
            vit.patch_embed.proj.weight.dtype = torch.float32 
            # 挂载 forward 行为
            vit.patch_embed.forward = MagicMock(side_effect=lambda x: torch.randn(100, 128))
            
            vit.fast_pos_embed_interpolate = MagicMock(return_value=torch.zeros(100, 128))
            vit.rot_pos_emb = MagicMock(return_value=torch.zeros(100, 64))
            vit.compute_attn_mask_seqlen = MagicMock(return_value=(100, [100]))

            mock_block = MockBlock() 
            vit.blocks = nn.ModuleList([mock_block, mock_block])

            mock_merger = MockBlock()
            mock_merger.forward = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 256))
            vit.deepstack_merger_list = nn.ModuleList([mock_merger, mock_merger])

            # [关键修复1] 使用 MockBlock
            vit.merger = MockBlock()
            vit.merger.forward = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 256))

            dummy_pixels = torch.randn(10, 3, 14, 14)
            grid_thw = [[1, 1, 1]] 
            
            output = vit.forward(dummy_pixels, grid_thw)
            assert output.shape[-1] == 256 * 3 


class TestQwen3VLModel:
    @pytest.fixture
    def mock_vllm_config(self):
        config = MagicMock() 
        model_config = MagicMock(spec=ModelConfig)
        model_config.hf_config = MagicMock()
        
        qwen_config = model_config.hf_config
        qwen_config.vision_config.deepstack_visual_indexes = [0, 1]
        qwen_config.vision_config.out_hidden_size = 256
        qwen_config.text_config.hidden_size = 1024
        qwen_config.image_token_id = 151652
        qwen_config.video_token_id = 151653
        
        mm_config = MagicMock()
        mm_config.mm_encoder_tp_mode = "data"
        mm_config.get_limit_per_prompt.return_value = 1
        model_config.multimodal_config = mm_config
        
        config.model_config = model_config
        config.quant_config = None
        config.scheduler_config = MagicMock()
        config.scheduler_config.max_num_batched_tokens = 2048
        
        return config

    def init_model(self, mock_vllm_config):
        with patch.object(Qwen3_VisionTransformer, "__init__", return_value=None), \
             patch.object(Qwen3LLMForCausalLM, "__init__", return_value=None):
            
            model = Qwen3VLForConditionalGeneration(vllm_config=mock_vllm_config)
            
            # [关键修复2] 手动调用 nn.Module 的初始化
            # 因为 __init__ 被 Mock 掉了，导致内部状态缺失，不能直接赋值子模块
            nn.Module.__init__(model.visual)
            nn.Module.__init__(model.language_model)
            
            model.visual.spatial_merge_size = 2
            model.visual.patch_embed = MockBlock()
            model.visual.patch_embed.proj = MagicMock()
            model.visual.patch_embed.proj.weight.dtype = torch.float16
            
            model.language_model.get_input_embeddings = MagicMock(
                side_effect=lambda input_ids: torch.randn(input_ids.shape[0], 1024)
            )
            # 这里 model.language_model 已经是初始化过的 Module，可以赋值
            model.language_model.model = MagicMock()

            model.use_deepstack = True
            model.deepstack_num_level = 2
            model.deepstack_input_embeds = [
                torch.zeros(2048, 1024),
                torch.zeros(2048, 1024)
            ]
            
            model.visual_dim = 1024
            model.multiscale_dim = 2048
            
            # 确保 Mock 返回维度与 split 逻辑匹配 (1024 + 2048 = 3072)
            def mock_process_image(image_input):
                return (torch.randn(100, 3072), ) 
            
            model._process_image_input = MagicMock(side_effect=mock_process_image)
            return model

    def test_init_deepstack_buffer(self, mock_vllm_config):
        with patch.object(Qwen3_VisionTransformer, "__init__", return_value=None), \
             patch.object(Qwen3LLMForCausalLM, "__init__", return_value=None):
             
             model = Qwen3VLForConditionalGeneration(vllm_config=mock_vllm_config)
             
             assert model.use_deepstack is True
             assert model.deepstack_num_level == 2
             assert model.deepstack_input_embeds is not None
             assert len(model.deepstack_input_embeds) == 2
             assert model.deepstack_input_embeds[0].shape == (2048, 1024)

    def test_get_input_embeddings_with_deepstack(self, mock_vllm_config):
        model = self.init_model(mock_vllm_config)
        
        def mock_merge(input_ids, inputs_embeds, multimodal_embeddings, placeholder_token_id):
            return inputs_embeds 
        
        with patch(target_merge_multimodal, side_effect=mock_merge) as merge_mock:
            input_ids = torch.randint(0, 1000, (50,))
            dummy_image_input = {"type": "pixel_values", "pixel_values": torch.randn(1, 3, 14, 14), "image_grid_thw": torch.tensor([[1, 1, 1]])}
            
            inputs_embeds = model.get_input_embeddings_v0(
                input_ids=input_ids,
                image_input=dummy_image_input
            )

            assert model._process_image_input.called
            assert merge_mock.call_count >= 2 

    def test_forward_deepstack_flow(self, mock_vllm_config):
        model = self.init_model(mock_vllm_config)
        
        with patch(target_get_pp_group, return_value=MagicMock(is_first_rank=True)):
            input_ids = torch.randint(0, 100, (10,))
            positions = torch.arange(10)
            inputs_embeds = torch.randn(10, 1024)
            model.deepstack_input_embeds = [torch.randn(2048, 1024) for _ in range(2)]
            model.language_model.model.return_value = torch.randn(10, 1024)

            model.forward(input_ids=None, positions=positions, inputs_embeds=inputs_embeds)
            
            args, kwargs = model.language_model.model.call_args
            assert "deepstack_input_embeds" in kwargs
            ds_args = kwargs["deepstack_input_embeds"]
            assert ds_args["deepstack_input_embeds_0"].shape == (10, 1024)
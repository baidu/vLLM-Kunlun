import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

# 根据实际文件路径调整导入
try:
    from vllm.model_executor.models.internvl import (
        InternVLChatModel,
        InternVisionModel,
        InternVLImagePixelInputs,
        InternVLVideoPixelInputs
    )
    # 用于 patch 的路径
    target_intern_vision = "vllm.model_executor.models.internvl.InternVisionModel"
    target_init_vllm_model = "vllm.model_executor.models.internvl.init_vllm_registered_model"
    target_merge = "vllm.model_executor.models.internvl.merge_multimodal_embeddings"
    target_auto_loader = "vllm.model_executor.models.internvl.AutoWeightsLoader"
except ImportError:
    # 兼容可能的路径差异 (vllm_kunlun)
    from vllm_kunlun.models.internvl import (
        InternVLChatModel,
        InternVisionModel,
        InternVLImagePixelInputs,
        InternVLVideoPixelInputs
    )
    target_intern_vision = "vllm_kunlun.models.internvl.InternVisionModel"
    target_init_vllm_model = "vllm_kunlun.models.internvl.init_vllm_registered_model"
    target_merge = "vllm_kunlun.models.internvl.merge_multimodal_embeddings"
    target_auto_loader = "vllm_kunlun.models.internvl.AutoWeightsLoader"

from vllm.config import VllmConfig, ModelConfig

# [修复点] 增强 MockModule，使其拥有 language_model 所需的方法
class MockModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # 这里的 MagicMock 充当 make_empty_intermediate_tensors 方法
        self.make_empty_intermediate_tensors = MagicMock()
        
    def forward(self, x, **kwargs):
        return x

class TestInternVLChatModel:
    
    @pytest.fixture
    def mock_vllm_config(self):
        config = MagicMock()
        
        # Model Config
        model_config = MagicMock(spec=ModelConfig)
        model_config.hf_config = MagicMock()
        model_config.multimodal_config = MagicMock()
        model_config.multimodal_config.mm_encoder_tp_mode = "data"
        
        # InternVL Config Structure
        hf_config = model_config.hf_config
        
        # Vision Config
        hf_config.vision_config.image_size = 448
        hf_config.vision_config.patch_size = 14
        hf_config.vision_config.hidden_size = 1024
        hf_config.vision_config.num_hidden_layers = 12
        hf_config.select_layer = -1
        
        # Global Config
        hf_config.force_image_size = None
        hf_config.downsample_ratio = 0.5 # 关键参数
        hf_config.ps_version = 'v2'
        
        # Text Config
        hf_config.text_config.hidden_size = 2048
        hf_config.text_config.architectures = ["InternLM2ForCausalLM"]
        
        config.model_config = model_config
        config.quant_config = None
        
        return config

    def init_model(self, mock_vllm_config):
        """
        初始化模型，Mock 掉所有繁重的子模块初始化
        """
        # Patch 掉 Vision Model 和 LLM 的初始化函数
        with patch(target_intern_vision, side_effect=MockModule), \
             patch(target_init_vllm_model, return_value=MockModule()):
            
            model = InternVLChatModel(vllm_config=mock_vllm_config)
            
            # 手动 Mock vision_model 的行为
            # 假设 Vision Model 输出 [B, N, H_vit]
            # 这里必须是一个 nn.Module (MockModule)，否则 setattr 会报错
            model.vision_model = MockModule()
            model.vision_model.forward = MagicMock(return_value=torch.randn(1, 257, 1024)) # 256 patches + 1 cls token
            
            # 手动 Mock MLP1 (Projector)
            # MLP1 负责将 Vision 维度投影到 LLM 维度
            # 输入维度取决于 downsample_ratio: hidden_size * (1/0.5)^2 = 1024 * 4 = 4096
            model.mlp1 = MockModule()
            model.mlp1.forward = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], x.shape[1], 2048)) # Output LLM dim
            
            # Mock Language Model
            # init 已经赋予了 MockModule，我们这里进一步配置它的方法
            # 注意：不能直接 model.language_model = MagicMock()，因为它是 Module 子模块
            
            # 配置 get_input_embeddings
            model.language_model.get_input_embeddings = MagicMock(
                side_effect=lambda input_ids: torch.randn(input_ids.shape[0], 2048)
            )
            # 配置 model() 方法 (即 forward)
            model.language_model.model = MagicMock()
            
            return model

    def test_init_attributes(self, mock_vllm_config):
        """测试初始化参数计算是否正确"""
        model = self.init_model(mock_vllm_config)
        
        assert model.patch_size == 14
        assert model.downsample_ratio == 0.5
        
        # num_image_token 计算验证:
        # (448 // 14)^2 * (0.5)^2 = 32^2 * 0.25 = 1024 * 0.25 = 256
        assert model.num_image_token == 256
        assert model.is_mono is False

    def test_pixel_shuffle_logic(self, mock_vllm_config):
        """测试核心的 Pixel Shuffle 逻辑"""
        model = self.init_model(mock_vllm_config)
        
        # 构造一个 Feature Map: [N, H, W, C]
        # 假设 H=32, W=32, C=1024 (Vision Hidden Size)
        # downsample_ratio = 0.5, 意味着 H, W 减半，C 变为 4 倍
        x = torch.randn(1, 32, 32, 1024)
        
        output = model.pixel_shuffle(x, scale_factor=0.5)
        
        # 验证形状
        # H_new = 32 * 0.5 = 16
        # W_new = 32 * 0.5 = 16
        # C_new = 1024 / (0.5 * 0.5) = 1024 * 4 = 4096
        assert output.shape == (1, 16, 16, 4096)

    def test_extract_feature(self, mock_vllm_config):
        """测试特征提取全流程: Vision -> Reshape -> PixelShuffle -> MLP"""
        model = self.init_model(mock_vllm_config)
        
        # 1. 模拟 Vision Model 输出
        # 输入 Pixel Values (shape 不关键，因为 vision_model 被 mock 了)
        pixel_values = torch.randn(2, 3, 448, 448) 
        
        # Vision Model 输出: [B, L+1, C] -> [2, 1025, 1024] (1024 patches + 1 cls)
        # 这里的 1024 patches 对应 32x32 的 grid
        vit_output = torch.randn(2, 1025, 1024)
        model.vision_model.forward.return_value = vit_output
        
        # 2. 执行 extract_feature
        features = model.extract_feature(pixel_values)
        
        # 3. 验证逻辑链
        # Step 1: Remove CLS token -> [2, 1024, 1024]
        # Step 2: Reshape -> [2, 32, 32, 1024]
        # Step 3: Pixel Shuffle (0.5) -> [2, 16, 16, 4096]
        # Step 4: Reshape -> [2, 256, 4096] (16*16=256)
        # Step 5: MLP -> [2, 256, 2048] (Project to LLM dim)
        
        assert features.shape == (2, 256, 2048)
        assert model.vision_model.forward.called
        assert model.mlp1.forward.called

    def test_process_image_input(self, mock_vllm_config):
        """测试图像输入处理与特征切分"""
        model = self.init_model(mock_vllm_config)
        
        # Mock extract_feature 以返回确定维度的 Tensor
        # 假设 batch 中有 2 张图，分别有 1 个 patch 和 2 个 patch
        # Total patches = 1 + 2 = 3
        # extract_feature 输入是 flattened pixel values
        # 输出是 [Total_Patches, Tokens_Per_Patch, Hidden]
        # Tokens_Per_Patch (after pixel shuffle) = 256
        
        # 模拟 extract_feature 返回所有 patches 的 embedding
        # 形状: [3, 256, 2048]
        model.extract_feature = MagicMock(return_value=torch.randn(3, 256, 2048))
        
        # 构造输入
        image_input = {
            "type": "pixel_values",
            "pixel_values_flat": MagicMock(), # Placeholder
            "num_patches": [1, 2] # 第一张图 1 块，第二张图 2 块
        }
        
        outputs = model._process_image_input(image_input)
        
        # 验证结果是一个 tuple，包含两个 tensor
        assert len(outputs) == 2
        
        # 第一张图: 1 patch * 256 tokens
        assert outputs[0].shape == (256, 2048)
        # 第二张图: 2 patches * 256 tokens = 512 tokens
        assert outputs[1].shape == (512, 2048)

    def test_get_multimodal_embeddings(self, mock_vllm_config):
        """测试多模态 Embedding 获取 (Image & Video)"""
        model = self.init_model(mock_vllm_config)
        
        # Mock _process_image_input，因为它是分别处理 image 和 video 的入口
        # 假设它返回 embedding list
        model._process_image_input = MagicMock(side_effect=[
            (torch.randn(256, 2048), ), # For images call
            (torch.randn(512, 2048), )  # For videos call
        ])
        
        # Mock _parse_and_validate
        # 这里模拟返回包含 image 和 video 的字典
        model._parse_and_validate_image_input = MagicMock(return_value="mock_img_input")
        model._parse_and_validate_video_input = MagicMock(return_value="mock_vid_input")
        
        # 调用
        embeddings = model.get_multimodal_embeddings(
            pixel_values_flat=MagicMock(), 
            pixel_values_flat_video=MagicMock()
        )
        
        # 验证
        # 应该包含 Image embedding 和 Video embedding
        assert len(embeddings) == 2
        assert embeddings[0].shape == (256, 2048) # Image
        assert embeddings[1].shape == (512, 2048) # Video
        
        # 验证调用顺序: 先处理 image, 再处理 video
        assert model._process_image_input.call_count == 2

    def test_forward_pass(self, mock_vllm_config):
        """测试 Forward 函数的参数透传"""
        model = self.init_model(mock_vllm_config)
        
        # Mock 依赖
        input_ids = torch.tensor([1, 2, 3])
        positions = torch.tensor([0, 1, 2])
        
        # 模拟 get_multimodal_embeddings 返回空 (测试纯文本路径或 embedding 已注入路径)
        model.get_multimodal_embeddings = MagicMock(return_value=[])
        
        # Mock merge
        with patch(target_merge) as mock_merge:
            mock_merge.return_value = torch.randn(3, 2048)
            
            # 设定 context token ids
            model.img_context_token_id = 100
            model.video_context_token_id = 101
            
            # 1. 测试有 visual embeddings 的情况
            # 手动注入 visual embeddings
            vision_emb = [torch.randn(256, 2048)]
            model.get_multimodal_embeddings.return_value = vision_emb
            
            model.forward(input_ids=input_ids, positions=positions)
            
            # 验证 merge 被调用
            assert mock_merge.called
            
            # 验证 LLM forward 被调用
            assert model.language_model.model.called
            args, kwargs = model.language_model.model.call_args
            assert "inputs_embeds" in kwargs
            # inputs_embeds 应该是 merge 的结果
            assert kwargs["inputs_embeds"].shape == (3, 2048)

    def test_load_weights(self, mock_vllm_config):
        """测试权重加载时的 skip logic"""
        model = self.init_model(mock_vllm_config)
        
        with patch(target_auto_loader) as MockLoader:
            loader_instance = MockLoader.return_value
            
            weights = [("vision_model.patch_embed.weight", torch.randn(1))]
            model.load_weights(weights)
            
            # 验证 AutoWeightsLoader 被初始化时传入了 skip_prefixes
            init_args, init_kwargs = MockLoader.call_args
            assert "skip_prefixes" in init_kwargs
            skips = init_kwargs["skip_prefixes"]
            
            # 验证 InternVL 特有的 skip key
            assert "temporal_embed" in skips
            assert "cg_model" in skips
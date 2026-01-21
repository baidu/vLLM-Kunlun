import sys
import os
import pytest
from unittest.mock import MagicMock, patch, ANY
from types import CodeType


from vllm_kunlun.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher

# ==========================================
# 辅助 Mock 和 Fixtures
# ==========================================

class ConcreteWrapper(TorchCompileWrapperWithCustomDispatcher):
    """
    具体的实现类，用于测试抽象基类。
    实现 forward 方法。
    """
    def forward(self, x):
        return x + 1

@pytest.fixture
def mock_vllm_config():
    """Mock vLLM 的配置对象"""
    with patch("vllm.config.get_current_vllm_config") as mock_get_cfg:
        mock_config = MagicMock()
        # 设置 compilation_config
        mock_config.compilation_config.init_backend.return_value = "inductor"
        mock_config.compilation_config.inductor_compile_config = {"options": "test"}
        mock_config.compilation_config.local_cache_dir = None # 默认不测试文件写入
        
        # 模拟 CompilationLevel 枚举
        with patch("vllm.config.CompilationLevel") as mock_level:
            mock_level.DYNAMO_ONCE = 1
            mock_get_cfg.return_value = mock_config
            yield mock_config

@pytest.fixture
def mock_torch_compile():
    """Mock torch.compile 以避免真实编译"""
    with patch("torch.compile") as mock_compile:
        # 让 compile 返回一个 mock 对象，可以直接调用
        mock_compile.return_value = MagicMock(return_value="compiled_result")
        yield mock_compile

@pytest.fixture
def mock_register_hook():
    """Mock 字节码 hook 注册"""
    with patch("torch._dynamo.convert_frame.register_bytecode_hook") as mock_hook:
        yield mock_hook

# ==========================================
# 测试用例
# ==========================================

def test_init_defaults(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """
    测试初始化：验证配置加载、torch.compile 调用和 hook 注册。
    """
    wrapper = ConcreteWrapper(compilation_level=0)
    
    # 验证是否获取了配置
    assert wrapper.vllm_config == mock_vllm_config
    
    # 验证是否调用了 torch.compile
    mock_torch_compile.assert_called_once()
    args, kwargs = mock_torch_compile.call_args
    # 第一个参数应该是 bound method forward
    assert args[0] == wrapper.forward
    assert kwargs['backend'] == "inductor"
    assert kwargs['fullgraph'] is True
    
    # 验证是否注册了 hook
    mock_register_hook.assert_called_once_with(wrapper.bytecode_hook)
    
    # 验证 dispatcher 开关逻辑 (level 0 < DYNAMO_ONCE 1)
    assert wrapper.use_custom_dispatcher is False

def test_init_with_custom_compilation_level(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """测试不同编译等级对 use_custom_dispatcher 的影响"""
    # 传入 level 2，应该大于 DYNAMO_ONCE (mock为1)
    wrapper = ConcreteWrapper(compilation_level=2)
    assert wrapper.use_custom_dispatcher is True

def test_call_method(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """
    测试 __call__ 方法：验证是否委托给了编译后的 callable。
    """
    wrapper = ConcreteWrapper()
    # 模拟编译后的函数被调用
    result = wrapper("input_data")
    
    # 验证结果
    assert result == "compiled_result"
    # 验证 compiled_callable 被调用
    wrapper.compiled_callable.assert_called_once_with("input_data")

def test_dispatch_to_code_context_manager(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """
    测试 dispatch_to_code 上下文管理器。
    核心逻辑：验证 forward.__code__ 在进入上下文时被替换，退出时被还原。
    """
    wrapper = ConcreteWrapper()
    
    # 创建两个假的 Code Object
    # 使用 compile() 内置函数快速生成 CodeType 对象
    original_code = wrapper.forward.__code__
    dummy_code_source = "def forward(self, x): return x * 2"
    dummy_code = compile(dummy_code_source, "<string>", "exec").co_consts[0]
    
    # 将假代码放入 compiled_codes 列表
    wrapper.compiled_codes.append(dummy_code)
    
    # 确保初始状态正确
    assert wrapper.__class__.forward.__code__ == original_code
    
    # 进入上下文
    with wrapper.dispatch_to_code(0):
        # 验证代码对象已被替换
        assert wrapper.__class__.forward.__code__ == dummy_code
        assert wrapper.__class__.forward.__code__ != original_code
    
    # 退出上下文，验证代码对象已还原
    assert wrapper.__class__.forward.__code__ == original_code

def test_bytecode_hook_success(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """
    测试 bytecode_hook：当条件满足时，应该保存 new_code。
    难点：需要模拟 sys._getframe 返回的堆栈结构。
    """
    wrapper = ConcreteWrapper()
    
    old_code = wrapper.original_code_object
    new_code = MagicMock(spec=CodeType)
    
    # --- 构造复杂的 Frame Mock ---
    # 逻辑路径：
    # 1. code_name == "_compile" AND file_name == "convert_frame.py"
    # 2. frame.f_locals["frame"].f_code == old_code
    # 3. frame.f_locals["frame"].f_locals["self"] is wrapper
    
    # 模拟最内层 frame (执行 forward 的 frame)
    inner_frame = MagicMock()
    inner_frame.f_code = old_code
    inner_frame.f_locals = {"self": wrapper}

    # 模拟 convert_frame.py 中的 _compile frame
    compile_frame = MagicMock()
    compile_frame.f_code.co_name = "_compile"
    compile_frame.f_code.co_filename = f"path{os.path.sep}convert_frame.py"
    # 这里的 f_locals["frame"] 对应源码中的 frame 变量，它指向 inner_frame
    compile_frame.f_locals = {"frame": inner_frame} 
    
    # 链接 frame 链 (mock sys._getframe().f_back...)
    # 我们可以直接 Patch sys._getframe 让它返回一个 dummy，然后 dummy.f_back 指向 compile_frame
    dummy_top_frame = MagicMock()
    dummy_top_frame.f_back = compile_frame
    
    with patch("sys._getframe", return_value=dummy_top_frame):
        # 执行 Hook
        wrapper.bytecode_hook(old_code, new_code)
    
    # 验证 new_code 被添加到了 compiled_codes
    assert new_code in wrapper.compiled_codes
    assert len(wrapper.compiled_codes) == 1

def test_bytecode_hook_ignored_code(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """测试 bytecode_hook：当 old_code 不匹配时，直接返回"""
    wrapper = ConcreteWrapper()
    
    random_code = MagicMock(spec=CodeType)
    new_code = MagicMock(spec=CodeType)
    
    # 此时 old_code != wrapper.original_code_object
    wrapper.bytecode_hook(random_code, new_code)
    
    # 应该没有任何动作
    assert len(wrapper.compiled_codes) == 0

def test_bytecode_hook_frame_mismatch(mock_vllm_config, mock_torch_compile, mock_register_hook):
    """测试 bytecode_hook：当 self 实例不匹配时，不保存"""
    wrapper = ConcreteWrapper()
    other_wrapper = ConcreteWrapper() # 另一个实例
    
    old_code = wrapper.original_code_object
    new_code = MagicMock(spec=CodeType)
    
    # 构造 Frame，但是 f_locals['self'] 是另一个对象
    inner_frame = MagicMock()
    inner_frame.f_code = old_code
    inner_frame.f_locals = {"self": other_wrapper} # <--- 不匹配

    compile_frame = MagicMock()
    compile_frame.f_code.co_name = "_compile"
    compile_frame.f_code.co_filename = f"path{os.path.sep}convert_frame.py"
    compile_frame.f_locals = {"frame": inner_frame}
    
    dummy_top_frame = MagicMock()
    dummy_top_frame.f_back = compile_frame
    
    with patch("sys._getframe", return_value=dummy_top_frame):
        wrapper.bytecode_hook(old_code, new_code)
        
    # 验证未添加
    assert len(wrapper.compiled_codes) == 0
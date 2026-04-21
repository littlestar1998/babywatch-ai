"""
核心模块包

【AI 初学者说明】
这个包包含了项目的核心功能模块：
- model_registry: 模型注册表，管理所有模型的元数据
- model_loader: 模型加载器，统一处理不同格式的模型加载
- inference_engine: 推理引擎，执行模型推理
- gpu_monitor: GPU 监控器，实时监控 GPU 使用情况
"""

from .model_registry import ModelRegistry, ModelType, ModelFormat, ModelMetadata
from .model_loader import ModelLoader
from .inference_engine import InferenceEngine, InferenceResult
from .gpu_monitor import GPUMonitor, GPUInfo

__all__ = [
    'ModelRegistry',
    'ModelType',
    'ModelFormat',
    'ModelMetadata',
    'ModelLoader',
    'InferenceEngine',
    'InferenceResult',
    'GPUMonitor',
    'GPUInfo',
]
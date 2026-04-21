"""
模型加载器 - 统一处理不同格式模型的加载

【AI 初学者说明】
不同的模型格式需要不同的加载方式：
- PyTorch (.pt): 使用 torch.load() 加载，需要完整的 PyTorch 环境
- ONNX (.onnx): 使用 ONNX Runtime 加载，兼容性好
- TensorRT (.engine): 使用 TensorRT API 加载，在 Jetson 上性能最佳

这个模块把这些差异封装起来，让调用者不需要关心具体细节。
就像一个"万能充电器"，不管是什么设备，都能自动匹配合适的接口。
"""

from typing import Any, Optional, Dict
from pathlib import Path
import time

from .model_registry import ModelFormat, ModelType


class ModelLoader:
    """
    统一的模型加载接口

    【AI 初学者说明】
    这个类使用"工厂模式"，根据模型格式自动选择合适的加载方法。
    就像一家工厂，你告诉它要什么产品，它自动选择对应的生产线。

    使用方法：
    ```python
    model = ModelLoader.load("models/yolov8n.pt", ModelFormat.PYTORCH)
    ```
    """

    @staticmethod
    def load(model_path: str, model_format: ModelFormat, model_type: ModelType = None) -> Any:
        """
        根据格式加载模型

        【AI 初学者说明】
        这是核心方法，根据模型格式自动选择加载方式：
        - PyTorch 格式 → 调用 PyTorch 加载器
        - ONNX 格式 → 调用 ONNX Runtime
        - TensorRT 格式 → 调用 TensorRT 引擎

        Args:
            model_path: 模型文件路径
            model_format: 模型格式（PYTORCH/ONNX/TENSORRT）
            model_type: 模型类型（可选，用于特殊处理）

        Returns:
            加载后的模型对象，不同格式返回不同类型的对象
        """
        # 验证文件存在
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if model_format == ModelFormat.PYTORCH:
            return ModelLoader._load_pytorch(model_path, model_type)
        elif model_format == ModelFormat.ONNX:
            return ModelLoader._load_onnx(model_path)
        elif model_format == ModelFormat.TENSORRT:
            return ModelLoader._load_tensorrt(model_path)
        elif model_format == ModelFormat.TFLITE:
            return ModelLoader._load_tflite(model_path)
        else:
            raise ValueError(f"不支持的模型格式: {model_format}")

    @staticmethod
    def _load_pytorch(model_path: str, model_type: Optional[ModelType] = None) -> Any:
        """
        加载 PyTorch 模型

        【AI 初学者说明】
        PyTorch 模型通常有两种形式：
        1. 完整模型（包含结构和权重）- 直接加载即可使用
        2. 仅权重（state_dict）- 需要先定义模型结构再加载权重

        这里我们优先尝试使用 Ultralytics YOLO 格式加载，
        因为 YOLO 是最常用的目标检测模型，格式友好。

        如果不是 YOLO 格式，则使用通用 PyTorch 加载方法。
        """
        import torch

        # 检测可用设备（GPU 优先）
        device = ModelLoader._get_device()
        print(f"[模型加载] 使用设备: {device}")

        # 尝试使用 Ultralytics YOLO 格式（常用目标检测模型）
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            # 设置模型到目标设备
            model.to(device)
            print(f"[模型加载] 成功加载 YOLO 模型: {model_path}")
            return model
        except ImportError:
            print("[模型加载] Ultralytics 未安装，使用通用 PyTorch 加载")
        except Exception as e:
            print(f"[模型加载] 不是 YOLO 格式: {e}")

        # 通用 PyTorch 加载
        try:
            checkpoint = torch.load(model_path, map_location=device)

            # 如果是 state_dict（仅权重），返回字典
            if isinstance(checkpoint, dict):
                print(f"[模型加载] 加载 PyTorch state_dict: {model_path}")
                return checkpoint

            # 如果是完整模型，设置为评估模式
            if hasattr(checkpoint, 'eval'):
                checkpoint.eval()
                print(f"[模型加载] 加载完整 PyTorch 模型: {model_path}")
            return checkpoint

        except Exception as e:
            raise RuntimeError(f"PyTorch 模型加载失败: {e}")

    @staticmethod
    def _load_onnx(model_path: str) -> Any:
        """
        加载 ONNX 模型

        【AI 初学者说明】
        ONNX（Open Neural Network Exchange）是一种通用的模型格式：
        - 可以由 PyTorch、TensorFlow 等框架导出
        - 使用 ONNX Runtime 进行推理
        - 支持多种执行后端（CPU、GPU、TensorRT）

        在 Jetson 上，我们优先使用 CUDA 执行提供者，
        这样可以利用 GPU 加速推理。
        """
        import onnxruntime as ort

        # 在 Jetson 上优先使用 CUDA 执行提供者
        available_providers = ort.get_available_providers()
        print(f"[模型加载] 可用的 ONNX 执行提供者: {available_providers}")

        # 优先级：CUDA > TensorRT > CPU
        providers_priority = ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CPUExecutionProvider']
        selected_providers = [p for p in providers_priority if p in available_providers]

        if not selected_providers:
            selected_providers = ['CPUExecutionProvider']

        print(f"[模型加载] 选择的执行提供者: {selected_providers}")

        try:
            session = ort.InferenceSession(model_path, providers=selected_providers)
            print(f"[模型加载] 成功加载 ONNX 模型: {model_path}")
            return session
        except Exception as e:
            raise RuntimeError(f"ONNX 模型加载失败: {e}")

    @staticmethod
    def _load_tensorrt(model_path: str) -> Any:
        """
        加载 TensorRT 引擎

        【AI 初学者说明】
        TensorRT 是 NVIDIA 的推理优化引擎：
        - 将模型优化为针对特定 GPU 的格式
        - 启动时需要反序列化引擎文件
        - 推理速度比普通 PyTorch 快 5-10 倍

        TensorRT 引擎的特点：
        - 只能在相同 GPU 架构上使用（不能跨设备）
        - 启动后创建执行上下文（context）
        - 推理时需要手动管理 GPU 内存

        注意：TensorRT 需要安装 tensorrt 和 pycuda 包
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT 加载需要安装 tensorrt 和 pycuda")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        try:
            # 读取引擎文件
            with open(model_path, 'rb') as f:
                engine_data = f.read()

            # 反序列化引擎
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(engine_data)

            # 创建执行上下文
            context = engine.create_execution_context()

            print(f"[模型加载] 成功加载 TensorRT 引擎: {model_path}")
            print(f"[模型加载] 引擎绑定数量: {engine.num_bindings}")

            return {
                'engine': engine,
                'context': context,
                'logger': TRT_LOGGER
            }

        except Exception as e:
            raise RuntimeError(f"TensorRT 引擎加载失败: {e}")

    @staticmethod
    def _load_tflite(model_path: str) -> Any:
        """
        加载 TensorFlow Lite 模型

        【AI 初学者说明】
        TensorFlow Lite 是轻量级的模型格式：
        - 专为移动和嵌入式设备设计
        - 在 Jetson 上可能不如 TensorRT 高效
        - 使用 TensorFlow Lite 解释器加载
        """
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            # 如果没有 tflite_runtime，尝试使用完整 tensorflow
            try:
                import tensorflow as tf
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                return interpreter
            except ImportError:
                raise ImportError("TensorFlow Lite 加载需要安装 tflite-runtime 或 tensorflow")

        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"[模型加载] 成功加载 TFLite 模型: {model_path}")
        return interpreter

    @staticmethod
    def _get_device():
        """
        检测并返回最佳计算设备

        【AI 初学者说明】
        AI 模型可以在 CPU 或 GPU 上运行：
        - CPU: 通用，但速度慢
        - GPU: 专用，但速度快（可达 10-100 倍）

        Jetson Orin Nano 有强大的 GPU，
        我们应该优先使用 GPU 进行推理。
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device('cuda')
        except ImportError:
            pass

        return 'cpu'

    @staticmethod
    def get_model_info(model_path: str) -> Dict[str, Any]:
        """
        获取模型基本信息（不加载模型）

        【AI 初学者说明】
        这个方法用于快速获取模型文件信息，
        用于在列表中显示，避免加载大模型耗费时间。

        返回：
        - file_size: 文件大小（字节）
        - modified_time: 最后修改时间
        - extension: 文件扩展名
        - format_guess: 推测的模型格式
        """
        path = Path(model_path)
        if not path.exists():
            return {}

        stat = path.stat()
        extension = path.suffix.lower()

        # 根据扩展名推测格式
        format_guess = {
            '.pt': 'PyTorch',
            '.pth': 'PyTorch',
            '.onnx': 'ONNX',
            '.engine': 'TensorRT',
            '.plan': 'TensorRT',
            '.tflite': 'TensorFlow Lite',
        }.get(extension, 'Unknown')

        return {
            'file_size': stat.st_size,
            'modified_time': stat.st_mtime,
            'extension': extension,
            'format_guess': format_guess,
            'file_name': path.name,
        }

    @staticmethod
    def get_input_output_info(model: Any, model_format: ModelFormat) -> Dict[str, Any]:
        """
        获取模型的输入输出信息

        【AI 初学者说明】
        了解模型的输入输出格式很重要：
        - 输入形状：模型需要什么尺寸的图片（如 640x640）
        - 输出形状：模型输出什么数据（如检测框列表）

        这些信息帮助我们正确地预处理和后处理数据。
        """
        info = {'inputs': [], 'outputs': []}

        if model_format == ModelFormat.ONNX:
            # ONNX 模型可以直接获取输入输出信息
            for inp in model.get_inputs():
                info['inputs'].append({
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                })
            for out in model.get_outputs():
                info['outputs'].append({
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                })

        elif model_format == ModelFormat.TENSORRT:
            engine = model['engine']
            for i in range(engine.num_bindings):
                name = engine.get_binding_name(i)
                shape = engine.get_binding_shape(i)
                is_input = engine.binding_is_input(i)
                if is_input:
                    info['inputs'].append({'name': name, 'shape': shape})
                else:
                    info['outputs'].append({'name': name, 'shape': shape})

        elif model_format == ModelFormat.PYTORCH:
            # PyTorch 模型需要运行一次才能确定形状
            # 这里只返回提示信息
            info['note'] = 'PyTorch 模型输入输出形状需要运行推理后确定'

        return info
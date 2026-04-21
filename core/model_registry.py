"""
模型注册表 - 管理模型元数据

【AI 初学者说明】
这个模块负责记录和管理所有上传的模型信息，就像一个"模型档案库"。
每个模型都有详细的"身份证"，记录它的类型、用途、文件位置等信息。
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
from typing import Optional, Dict, Any, List


class ModelType(Enum):
    """
    模型类型枚举 - 定义模型的功能类别

    【AI 初学者说明】
    不同的模型类型处理不同的任务：
    - 目标检测：在图片中找到物体（如宝宝、玩具），画出边界框
    - 姿态估计：检测人体的关键点位置（如手、脚、头），用于判断姿势
    - 音频分类：分析声音，判断是哭声、笑声还是其他
    - 人脸关键点：检测面部的特征点，用于判断口鼻是否被遮挡
    """
    DETECTION = "目标检测"        # 检测图像中的物体（如宝宝、玩具等）
    POSE = "姿态估计"            # 检测人体关键点（如宝宝姿势）
    AUDIO = "音频分类"           # 分类音频（如哭声检测）
    FACE_LANDMARK = "人脸关键点"  # 检测面部关键点（如口鼻遮挡检测）
    SEGMENTATION = "图像分割"     # 分割图像区域
    CLASSIFICATION = "图像分类"   # 分类图像内容
    UNKNOWN = "未知"


class ModelFormat(Enum):
    """
    模型格式枚举 - 不同格式代表不同的推理引擎

    【AI 初学者说明】
    AI 模型有多种存储格式，各有优缺点：
    - PyTorch (.pt): 最通用的格式，但推理速度较慢
    - ONNX (.onnx): 通用中间格式，兼容性好，速度适中
    - TensorRT (.engine): NVIDIA 专用格式，在 Jetson 上最快（3-10倍提速）

    选择建议：
    - 开发阶段用 PyTorch，方便调试
    - 生产环境用 TensorRT，追求速度
    """
    PYTORCH = "PyTorch (.pt)"        # PyTorch 原生格式，通用但较慢
    ONNX = "ONNX (.onnx)"            # 通用中间格式，兼容性好
    TENSORRT = "TensorRT (.engine)"  # NVIDIA 专用格式，在 Jetson 上最快
    TFLITE = "TensorFlow Lite (.tflite)"  # 轻量级 TensorFlow 格式


@dataclass
class ModelMetadata:
    """
    模型元数据 - 模型的"身份证"

    【AI 初学者说明】
    元数据就是"关于数据的数据"，这里指描述模型的基本信息。
    使用 @dataclass 装饰器可以自动生成初始化和比较方法，简化代码。

    每个模型上传后，系统会自动创建这个"身份证"，记录：
    - 模型名称和唯一 ID
    - 文件路径和大小
    - 模型类型和格式
    - 上传时间
    - 用户添加的描述
    """
    model_id: str                    # 唯一标识符（自动生成）
    name: str                        # 显示名称（用户指定）
    file_path: str                   # 文件存储路径
    model_type: ModelType            # 模型类型（如目标检测）
    model_format: ModelFormat        # 模型格式（如 PyTorch）
    file_size: int                   # 文件大小（字节）
    upload_time: datetime            # 上传时间
    description: str = ""            # 用户描述
    input_shape: Optional[tuple] = None   # 输入尺寸，如 (640, 640)
    num_classes: Optional[int] = None      # 分类数量
    labels: List[str] = field(default_factory=list)  # 类别标签列表
    is_tensorrt_optimized: bool = False     # 是否已 TensorRT 优化
    original_model_path: Optional[str] = None  # 原 PyTorch 模型路径（用于 TensorRT）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于 JSON 存储"""
        d = asdict(self)
        d['model_type'] = self.model_type.name
        d['model_format'] = self.model_format.name
        d['upload_time'] = self.upload_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建，用于 JSON 加载"""
        d['model_type'] = ModelType[d['model_type']]
        d['model_format'] = ModelFormat[d['model_format']]
        d['upload_time'] = datetime.fromisoformat(d['upload_time'])
        return cls(**d)


class ModelRegistry:
    """
    模型注册表 - 管理所有模型的元数据

    【AI 初学者说明】
    这个类负责：
    1. 记录所有上传模型的信息（注册）
    2. 查询模型信息（查询）
    3. 删除模型记录（注销）

    使用 JSON 文件持久化存储，即使程序重启也能记住之前的模型。
    这就像一个数据库，专门存储模型的"档案"。
    """

    REGISTRY_FILE = "models/registry.json"

    def __init__(self):
        """初始化注册表，加载已有数据"""
        self._models: Dict[str, ModelMetadata] = {}
        self._load_registry()

    def register(self, metadata: ModelMetadata) -> None:
        """
        注册新模型

        【AI 初学者说明】
        当用户上传新模型时，系统会：
        1. 创建模型的"身份证"（ModelMetadata）
        2. 调用此方法将其记录到注册表
        3. 保存到 JSON 文件，永久存储
        """
        self._models[metadata.model_id] = metadata
        self._save_registry()

    def unregister(self, model_id: str) -> bool:
        """
        注销模型

        【AI 初学者说明】
        当用户删除模型时，系统会：
        1. 从注册表中移除该模型的记录
        2. 保存更新后的注册表

        返回 True 表示成功删除，False 表示找不到该模型
        """
        if model_id in self._models:
            del self._models[model_id]
            self._save_registry()
            return True
        return False

    def get(self, model_id: str) -> Optional[ModelMetadata]:
        """获取指定模型的元数据"""
        return self._models.get(model_id)

    def list_all(self) -> List[ModelMetadata]:
        """列出所有已注册的模型"""
        return list(self._models.values())

    def list_by_type(self, model_type: ModelType) -> List[ModelMetadata]:
        """按类型筛选模型"""
        return [m for m in self._models.values() if m.model_type == model_type]

    def list_by_format(self, model_format: ModelFormat) -> List[ModelMetadata]:
        """按格式筛选模型"""
        return [m for m in self._models.values() if m.model_format == model_format]

    def count(self) -> int:
        """获取模型总数"""
        return len(self._models)

    def count_by_type(self) -> Dict[str, int]:
        """统计各类型模型数量"""
        counts = {}
        for model in self._models.values():
            type_name = model.model_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts

    def _load_registry(self) -> None:
        """
        从 JSON 文件加载注册表

        【AI 初学者说明】
        程序启动时会自动调用此方法：
        1. 检查是否存在 registry.json 文件
        2. 如果存在，读取并解析 JSON 内容
        3. 将 JSON 数据转换为 ModelMetadata 对象
        4. 如果不存在，创建空注册表
        """
        registry_path = Path(self.REGISTRY_FILE)

        if not registry_path.exists():
            # 创建空的注册表文件
            self._ensure_registry_file()
            return

        try:
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for model_id, model_dict in data.get('models', {}).items():
                try:
                    self._models[model_id] = ModelMetadata.from_dict(model_dict)
                except Exception as e:
                    print(f"加载模型 {model_id} 时出错: {e}")

        except Exception as e:
            print(f"加载注册表时出错: {e}")

    def _save_registry(self) -> None:
        """
        保存注册表到 JSON 文件

        【AI 初学者说明】
        每次注册或注销模型后，都会调用此方法：
        1. 将所有 ModelMetadata 转换为字典
        2. 包装为 JSON 格式
        3. 写入 registry.json 文件
        """
        self._ensure_registry_file()

        data = {
            'version': '1.0',
            'models': {
                model_id: model.to_dict()
                for model_id, model in self._models.items()
            }
        }

        registry_path = Path(self.REGISTRY_FILE)
        with open(registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _ensure_registry_file(self) -> None:
        """确保注册表文件存在"""
        registry_path = Path(self.REGISTRY_FILE)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        if not registry_path.exists():
            # 创建空的注册表文件
            with open(registry_path, 'w', encoding='utf-8') as f:
                json.dump({'version': '1.0', 'models': {}}, f)
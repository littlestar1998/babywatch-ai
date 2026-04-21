"""
推理引擎 - 执行模型推理并返回结果

【AI 初学者说明】
"推理"是指使用训练好的模型对新数据进行预测的过程。
与"训练"不同，推理不需要更新模型参数，只需要前向传播。

推理流程：
1. 预处理：将输入数据（图片/音频）转换为模型能理解的格式
2. 执行推理：将预处理后的数据传入模型
3. 后处理：将模型输出转换为人类可理解的结果

这就像一个翻译过程：
- 预处理：把人类语言翻译成机器语言
- 推理：机器思考并给出答案
- 后处理：把机器答案翻译回人类语言
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List, Dict
import time
import numpy as np
import cv2

from .model_registry import ModelType, ModelFormat


@dataclass
class InferenceResult:
    """
    推理结果数据类

    【AI 初学者说明】
    使用数据类（dataclass）可以清晰地定义返回数据的结构。
    这比使用字典更安全，IDE 也能提供代码补全。

    每次推理都会返回这个结构，包含：
    - 是否成功
    - 原始输出和处理后的输出
    - 各阶段的耗时
    - GPU 显存占用
    """
    success: bool                    # 推理是否成功
    output: Any                      # 原始输出（模型返回值）
    processed_output: Any            # 处理后的结果（如检测框、关键点等）
    inference_time_ms: float         # 推理耗时（毫秒）
    preprocess_time_ms: float        # 预处理耗时
    postprocess_time_ms: float       # 后处理耗时
    total_time_ms: float             # 总耗时
    gpu_memory_used_mb: float        # GPU 显存占用
    error_message: Optional[str] = None  # 错误信息
    detections: List[Dict] = field(default_factory=list)  # 检测结果列表
    keypoints: List[List] = field(default_factory=list)   # 关键点列表
    classifications: Dict[str, float] = field(default_factory=dict)  # 分类结果


class InferenceEngine:
    """
    推理引擎 - 统一的推理接口

    【AI 初学者说明】
    这个类封装了不同类型模型的推理逻辑：
    - 目标检测模型：返回检测框和类别
    - 姿态估计模型：返回人体关键点
    - 音频分类模型：返回分类结果

    使用方法：
    ```python
    engine = InferenceEngine(model, ModelType.DETECTION, ModelFormat.PYTORCH)
    result = engine.run(image)
    ```
    """

    def __init__(self, model: Any, model_type: ModelType, model_format: ModelFormat):
        """
        初始化推理引擎

        Args:
            model: 已加载的模型对象
            model_type: 模型类型
            model_format: 模型格式
        """
        self.model = model
        self.model_type = model_type
        self.model_format = model_format
        self._device = self._detect_device()
        self._original_size = None  # 用于后处理时恢复原始尺寸

    def _detect_device(self):
        """检测可用设备（GPU/CPU）"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.device('cuda')
        except ImportError:
            pass
        return 'cpu'

    def run(self, input_data: Any, conf_threshold: float = 0.25) -> InferenceResult:
        """
        执行推理

        【AI 初学者说明】
        这是最核心的方法，根据模型类型自动选择推理流程：
        1. 记录开始时间
        2. 预处理输入数据
        3. 执行模型推理
        4. 后处理输出结果
        5. 记录结束时间，计算耗时

        Args:
            input_data: 输入数据（图片/音频等）
            conf_threshold: 置信度阈值（只保留置信度高于此值的结果）

        Returns:
            InferenceResult: 推理结果对象
        """
        gpu_before = self._get_gpu_memory()
        start_total = time.perf_counter()

        try:
            # 1. 预处理
            start_pre = time.perf_counter()
            preprocessed = self._preprocess(input_data)
            preprocess_time = (time.perf_counter() - start_pre) * 1000

            # 2. 推理
            start_inf = time.perf_counter()
            raw_output = self._inference(preprocessed, conf_threshold)
            inference_time = (time.perf_counter() - start_inf) * 1000

            # 3. 后处理
            start_post = time.perf_counter()
            processed_output, detections, keypoints, classifications = self._postprocess(
                raw_output, input_data, conf_threshold
            )
            postprocess_time = (time.perf_counter() - start_post) * 1000

            total_time = (time.perf_counter() - start_total) * 1000
            gpu_after = self._get_gpu_memory()

            return InferenceResult(
                success=True,
                output=raw_output,
                processed_output=processed_output,
                inference_time_ms=inference_time,
                preprocess_time_ms=preprocess_time,
                postprocess_time_ms=postprocess_time,
                total_time_ms=total_time,
                gpu_memory_used_mb=gpu_after - gpu_before,
                detections=detections,
                keypoints=keypoints,
                classifications=classifications
            )

        except Exception as e:
            total_time = (time.perf_counter() - start_total) * 1000
            return InferenceResult(
                success=False,
                output=None,
                processed_output=None,
                inference_time_ms=0,
                preprocess_time_ms=0,
                postprocess_time_ms=0,
                total_time_ms=total_time,
                gpu_memory_used_mb=0,
                error_message=str(e)
            )

    def _preprocess(self, input_data: Any) -> Any:
        """
        预处理输入数据

        【AI 初学者说明】
        不同类型的模型需要不同的预处理：
        - 图像模型：调整尺寸、颜色转换、归一化
        - 音频模型：提取特征（如梅尔频谱图）
        - 视频模型：逐帧处理

        预处理的目标是将原始数据转换为模型期望的格式。
        """
        if isinstance(input_data, np.ndarray):
            # 图像数据
            if len(input_data.shape) == 3:  # RGB/BGR 图像
                return self._preprocess_image(input_data)
            elif len(input_data.shape) == 2:  # 灰度图
                return self._preprocess_image(input_data)

        # 其他类型直接返回
        return input_data

    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
        """
        图像预处理

        【AI 初学者说明】
        AI 模型通常需要固定尺寸的输入，预处理包括：
        1. 保存原始尺寸（用于后处理时恢复坐标）
        2. 调整图像尺寸（resize）
        3. 颜色空间转换（如 BGR -> RGB）
        4. 归一化（将像素值从 0-255 缩放到 0-1）
        5. 调整维度顺序（HWC -> CHW，高度-宽度-通道 改为 通道-高度-宽度）

        这些步骤是标准的图像预处理流程，大多数视觉模型都需要。
        """
        # 保存原始尺寸用于后处理
        self._original_size = (image.shape[1], image.shape[0])  # (width, height)

        # 对于 Ultralytics YOLO，预处理由模型内部完成
        # 直接返回原始图像
        if self.model_format == ModelFormat.PYTORCH and hasattr(self.model, 'predict'):
            return image

        # 通用预处理流程
        # 1. 调整尺寸
        img = cv2.resize(image, target_size)

        # 2. BGR -> RGB（OpenCV 默认是 BGR，模型通常期望 RGB）
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 归一化到 0-1
        img = img.astype(np.float32) / 255.0

        # 4. HWC -> CHW（numpy 转置）
        img = img.transpose(2, 0, 1)

        # 5. 添加 batch 维度 (1, C, H, W)
        img = np.expand_dims(img, axis=0)

        return img

    def _inference(self, input_data: Any, conf_threshold: float) -> Any:
        """
        执行模型推理

        【AI 初学者说明】
        这一步将预处理后的数据传入模型，获取输出。
        不同格式的模型有不同的推理方式：

        - PyTorch: model(input) 或 model.predict(input)
        - ONNX: session.run(None, {input_name: input})
        - TensorRT: 需要手动管理 GPU 内存

        对于 Ultralytics YOLO，它封装了预处理和后处理，
        直接调用 predict() 即可。
        """
        if self.model_format == ModelFormat.PYTORCH:
            # Ultralytics YOLO 格式
            if hasattr(self.model, 'predict'):
                # YOLO 的 predict 方法包含了预处理和后处理
                results = self.model.predict(
                    input_data,
                    conf=conf_threshold,
                    verbose=False,
                    device=self._device
                )
                return results

            # 通用 PyTorch 推理
            import torch
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_tensor = torch.from_numpy(input_data).to(self._device)
                else:
                    input_tensor = input_data
                return self.model(input_tensor)

        elif self.model_format == ModelFormat.ONNX:
            # ONNX Runtime 推理
            input_name = self.model.get_inputs()[0].name
            return self.model.run(None, {input_name: input_data})

        elif self.model_format == ModelFormat.TENSORRT:
            # TensorRT 推理
            return self._tensorrt_inference(input_data)

        return None

    def _tensorrt_inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        TensorRT 推理

        【AI 初学者说明】
        TensorRT 推理比 PyTorch 复杂，需要手动管理内存：
        1. 分配 GPU 输入内存
        2. 将数据从 CPU 复制到 GPU
        3. 执行推理
        4. 将结果从 GPU 复制回 CPU
        5. 释放 GPU 内存

        这些步骤虽然繁琐，但换来的是显著的性能提升。
        """
        import pycuda.driver as cuda
        import pycuda.autoinit
        import numpy as np

        engine = self.model['engine']
        context = self.model['context']

        # 获取输入输出信息
        input_binding = 0
        output_binding = 1

        input_shape = engine.get_binding_shape(input_binding)
        output_shape = engine.get_binding_shape(output_binding)

        # 确保输入数据形状正确
        input_data = np.ascontiguousarray(input_data, dtype=np.float32)

        # 分配 GPU 内存
        input_size = int(np.prod(input_shape))
        output_size = int(np.prod(output_shape))

        # 创建输入输出缓冲区
        input_buffer = cuda.mem_alloc(input_size * input_data.itemsize)
        output_buffer = cuda.mem_alloc(output_size * input_data.itemsize)

        # 创建输出数组
        output_data = np.empty(output_shape, dtype=np.float32)

        # 将输入数据复制到 GPU
        cuda.memcpy_htod(input_buffer, input_data)

        # 执行推理
        context.set_binding_shape(input_binding, input_shape)
        context.execute_v2([int(input_buffer), int(output_buffer)])

        # 将输出数据复制回 CPU
        cuda.memcpy_dtoh(output_data, output_buffer)

        return output_data

    def _postprocess(
        self,
        raw_output: Any,
        original_input: Any,
        conf_threshold: float
    ) -> Tuple[Any, List[Dict], List[List], Dict[str, float]]:
        """
        后处理模型输出

        【AI 初学者说明】
        模型的原始输出通常是数字矩阵，需要转换为可理解的结果：
        - 目标检测：将输出矩阵转换为检测框列表（位置、类别、置信度）
        - 姿态估计：将输出转换为关键点坐标列表
        - 分类：将输出转换为类别概率分布

        后处理是推理流程的最后一步，决定了结果的可读性。
        """
        detections = []
        keypoints = []
        classifications = {}

        if self.model_type == ModelType.DETECTION:
            processed, detections = self._postprocess_detection(raw_output, conf_threshold)

        elif self.model_type == ModelType.POSE:
            processed, keypoints = self._postprocess_pose(raw_output, conf_threshold)

        elif self.model_type == ModelType.FACE_LANDMARK:
            processed, keypoints = self._postprocess_face_landmark(raw_output)

        elif self.model_type == ModelType.AUDIO:
            processed, classifications = self._postprocess_audio(raw_output)

        elif self.model_type == ModelType.CLASSIFICATION:
            processed, classifications = self._postprocess_classification(raw_output)

        else:
            processed = raw_output

        return processed, detections, keypoints, classifications

    def _postprocess_detection(self, raw_output: Any, conf_threshold: float) -> Tuple[Any, List[Dict]]:
        """
        后处理目标检测结果

        【AI 初学者说明】
        目标检测模型的输出通常是：
        - 检测框坐标：[x1, y1, x2, y2]（左上角和右下角）
        - 置信度：这个框确实有物体的概率
        - 类别：框内物体的类型（如"宝宝"、"玩具"）

        Ultralytics YOLO 已经做了大部分后处理，
        我们只需要提取检测结果即可。
        """
        detections = []

        # Ultralytics YOLO 格式
        if hasattr(raw_output, '__iter__') and len(raw_output) > 0:
            first_result = raw_output[0]
            if hasattr(first_result, 'boxes'):
                boxes = first_result.boxes

                # 将坐标恢复到原始图像尺寸
                for i in range(len(boxes)):
                    box = boxes[i]

                    # 获取坐标（YOLO 格式是 xyxy）
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # 恢复到原始尺寸（如果之前做了 resize）
                    if self._original_size:
                        orig_w, orig_h = self._original_size
                        # 假设预处理时 resize 到了 640
                        scale_x = orig_w / 640
                        scale_y = orig_h / 640
                        x1 *= scale_x
                        y1 *= scale_y
                        x2 *= scale_x
                        y2 *= scale_y

                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())

                    # 获取类别名称
                    class_name = first_result.names.get(cls, f"class_{cls}")

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name,
                        'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)]
                    })

        return raw_output, detections

    def _postprocess_pose(self, raw_output: Any, conf_threshold: float) -> Tuple[Any, List[List]]:
        """
        后处理姿态估计结果

        【AI 初学者说明】
        姿态估计模型检测人体的关键点位置：
        - 每个关键点有 (x, y, confidence) 三个值
        - 通常有 17 个关键点（COCO 标准）：头、肩膀、手肘、手腕、臀、膝盖、脚踝等

        关键点数据可用于判断宝宝的姿势：
        - 睡眠检测：关键点的分布和变化模式
        - 动作识别：关键点的时序变化
        """
        keypoints = []

        # Ultralytics YOLO Pose 格式
        if hasattr(raw_output, '__iter__') and len(raw_output) > 0:
            first_result = raw_output[0]
            if hasattr(first_result, 'keypoints'):
                kpts = first_result.keypoints

                if hasattr(kpts, 'data'):
                    kpts_data = kpts.data[0].tolist()  # [keypoint_id][x, y, conf]

                    for kp in kpts_data:
                        x, y, conf = kp
                        keypoints.append([x, y, conf])

        return raw_output, keypoints

    def _postprocess_face_landmark(self, raw_output: Any) -> Tuple[Any, List[List]]:
        """
        后处理人脸关键点结果

        【AI 初学者说明】
        人脸关键点模型检测面部的特征点：
        - MediaPipe Face Mesh 有 468 个关键点
        - 可以精确判断眼睛、鼻子、嘴巴的状态

        用于口鼻遮挡检测：
        - 检查鼻子和嘴巴区域的关键点是否可见
        - 如果关键点置信度低，说明可能被遮挡
        """
        keypoints = []
        # 具体实现取决于使用的模型
        return raw_output, keypoints

    def _postprocess_audio(self, raw_output: Any) -> Tuple[Any, Dict[str, float]]:
        """
        后处理音频分类结果

        【AI 初学者说明】
        音频分类模型输出声音的类别概率：
        - 如"哭声"、"笑声"、"说话声"的概率

        用于哭声检测：
        - 当"哭声"类别概率超过阈值时触发警报
        """
        classifications = {}

        # 将输出转换为概率分布
        if isinstance(raw_output, np.ndarray):
            # 假设输出是 logits，需要转换为概率
            probs = self._softmax(raw_output)

            # 根据模型类别映射
            # 这里需要根据具体模型的类别列表调整
            class_names = ['哭声', '笑声', '说话声', '安静', '其他']
            for i, prob in enumerate(probs[0]):
                if i < len(class_names):
                    classifications[class_names[i]] = prob

        return raw_output, classifications

    def _postprocess_classification(self, raw_output: Any) -> Tuple[Any, Dict[str, float]]:
        """
        后处理图像分类结果

        【AI 初学者说明】
        图像分类模型输出图像内容的类别概率：
        - 如"宝宝睡觉"、"宝宝玩耍"的概率
        """
        classifications = {}

        if isinstance(raw_output, np.ndarray):
            probs = self._softmax(raw_output)
            # 具体类别映射取决于模型
            classifications['top_class'] = np.argmax(probs)
            classifications['confidence'] = probs[0][np.argmax(probs)]

        return raw_output, classifications

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax 函数 - 将 logits 转换为概率

        【AI 初学者说明】
        神经网络的输出通常是 logits（未归一化的值），
        需要通过 softmax 转换为概率分布：
        - 所有概率值都在 0-1 之间
        - 所有概率之和为 1

        公式：softmax(x) = exp(x) / sum(exp(x))
        """
        exp_x = np.exp(x - np.max(x))  # 减去最大值防止数值溢出
        return exp_x / exp_x.sum()

    def _get_gpu_memory(self) -> float:
        """
        获取当前 GPU 显存使用量

        【AI 初学者说明】
        监控 GPU 显存很重要：
        - Jetson 使用统一内存，内存有限
        - 大模型可能占用大量显存
        - 显存不足会导致推理失败

        返回：当前已分配的显存（MB）
        """
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass
        return 0.0

    def visualize_result(
        self,
        image: np.ndarray,
        result: InferenceResult,
        draw_boxes: bool = True,
        draw_keypoints: bool = True
    ) -> np.ndarray:
        """
        在图像上可视化推理结果

        【AI 初学者说明】
        将检测结果绘制到图像上：
        - 检测框：矩形框，标注类别和置信度
        - 关键点：圆点，标注位置

        返回：绘制后的图像（numpy 数组）
        """
        output_image = image.copy()

        # 绘制检测框
        if draw_boxes and result.detections:
            for det in result.detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_name = det['class_name']

                # 绘制框（绿色）
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制标签背景
                label = f"{class_name}: {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(output_image, (x1, y1 - text_size[1] - 5),
                              (x1 + text_size[0], y1), (0, 255, 0), -1)

                # 绘制标签文字
                cv2.putText(output_image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 绘制关键点
        if draw_keypoints and result.keypoints:
            for kp in result.keypoints:
                x, y, conf = kp
                if conf > 0.5:  # 只绘制置信度高的关键点
                    cv2.circle(output_image, (int(x), int(y)), 5, (0, 0, 255), -1)

        return output_image
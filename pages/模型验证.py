"""
模型验证页面 - 测试模型效果
"""

import streamlit as st
import numpy as np
from pathlib import Path
import cv2
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_registry import ModelRegistry, ModelType, ModelFormat, ModelMetadata
from core.model_loader import ModelLoader
from core.inference_engine import InferenceEngine, InferenceResult


@st.cache_resource
def get_registry():
    return ModelRegistry()


@st.cache_resource
def load_model_cached(model_path: str, model_format: ModelFormat, model_type: ModelType):
    return ModelLoader.load(model_path, model_format, model_type)


def render_model_selector():
    """模型选择器"""
    registry = get_registry()
    models = registry.list_all()

    if not models:
        st.warning("暂无模型，请先上传")
        return None

    # 按类型分组
    models_by_type = {}
    for model in models:
        type_name = model.model_type.value
        if type_name not in models_by_type:
            models_by_type[type_name] = []
        models_by_type[type_name].append(model)

    # 类型选择
    type_options = list(models_by_type.keys())
    selected_type = st.selectbox("模型类型", type_options)

    # 模型选择
    type_models = models_by_type[selected_type]
    model_options = {m.name: m for m in type_models}
    selected_name = st.selectbox("选择模型", list(model_options.keys()))

    selected_model = model_options[selected_name]

    # 模型信息
    st.info(f"{selected_model.model_format.value.split()[0]} | {selected_model.file_size/(1024*1024):.1f} MB")

    if selected_model.description:
        st.caption(selected_model.description)

    return selected_model


def render_test_section(model: ModelMetadata):
    """测试区域"""

    # 置信度阈值
    conf_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.25, 0.05)

    # 根据模型类型显示上传界面
    if model.model_type in [ModelType.DETECTION, ModelType.POSE, ModelType.FACE_LANDMARK, ModelType.SEGMENTATION, ModelType.CLASSIFICATION]:
        test_media = render_image_uploader()
    elif model.model_type == ModelType.AUDIO:
        test_media = render_audio_uploader()
    else:
        test_media = None
        st.warning("该模型类型暂不支持")

    # 运行推理
    if test_media is not None:
        if st.button("运行推理", type="primary"):
            with st.spinner("推理中..."):
                result = run_inference(model, test_media, conf_threshold)
            render_result(model, test_media, result)


def render_image_uploader():
    """图像上传器"""
    uploaded_file = st.file_uploader("选择图片", type=['jpg', 'jpeg', 'png', 'bmp'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        st.caption(f"{image.size[0]} x {image.size[1]}")

        image_array = np.array(image)
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        return {'type': 'image', 'data': image_array, 'original': np.array(image)}

    return None


def render_audio_uploader():
    """音频上传器"""
    uploaded_file = st.file_uploader("选择音频", type=['wav', 'mp3', 'ogg'])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        return {'type': 'audio', 'data': uploaded_file}

    return None


def run_inference(model_metadata: ModelMetadata, test_media: dict, conf_threshold: float) -> InferenceResult:
    """运行推理"""
    try:
        loaded_model = load_model_cached(
            model_metadata.file_path,
            model_metadata.model_format,
            model_metadata.model_type
        )
    except Exception as e:
        return InferenceResult(
            success=False, output=None, processed_output=None,
            inference_time_ms=0, preprocess_time_ms=0, postprocess_time_ms=0,
            total_time_ms=0, gpu_memory_used_mb=0, error_message=str(e)
        )

    engine = InferenceEngine(loaded_model, model_metadata.model_type, model_metadata.model_format)
    return engine.run(test_media['data'], conf_threshold)


def render_result(model: ModelMetadata, test_media: dict, result: InferenceResult):
    """渲染结果"""
    st.markdown("---")
    st.subheader("推理结果")

    if not result.success:
        st.error(f"失败: {result.error_message}")
        return

    # 性能指标
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总耗时", f"{result.total_time_ms:.1f} ms")
    col2.metric("推理时间", f"{result.inference_time_ms:.1f} ms")
    col3.metric("预处理", f"{result.preprocess_time_ms:.1f} ms")
    col4.metric("显存", f"{result.gpu_memory_used_mb:.1f} MB")

    # 性能评估
    if result.total_time_ms < 50:
        st.success("优秀，适合实时应用")
    elif result.total_time_ms < 100:
        st.info("良好")
    elif result.total_time_ms < 200:
        st.warning("较慢，建议 TensorRT")
    else:
        st.error("过慢，不适合实时")

    # 结果可视化
    if model.model_type == ModelType.DETECTION:
        render_detection_result(test_media, result)
    elif model.model_type == ModelType.POSE:
        render_pose_result(test_media, result)
    elif model.model_type == ModelType.FACE_LANDMARK:
        render_face_landmark_result(test_media, result)
    elif model.model_type == ModelType.AUDIO:
        render_audio_result(result)
    elif model.model_type == ModelType.CLASSIFICATION:
        render_classification_result(result)


def render_detection_result(test_media: dict, result: InferenceResult):
    """目标检测结果"""
    st.markdown("**检测结果**")

    if test_media['type'] == 'image':
        image = test_media['original'].copy()

        for det in result.detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['class_name']
            conf = det['confidence']

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f"{label}: {conf:.2f}"
            cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        st.image(image, use_column_width=True)

    if result.detections:
        st.info(f"检测到 {len(result.detections)} 个目标")
        table_data = []
        for i, det in enumerate(result.detections):
            x1, y1, x2, y2 = det['bbox']
            table_data.append({
                '类别': det['class_name'],
                '置信度': f"{det['confidence']:.2%}",
                '位置': f"({x1},{y1})-({x2},{y2})"
            })
        st.dataframe(table_data, hide_index=True)
    else:
        st.warning("未检测到目标")


def render_pose_result(test_media: dict, result: InferenceResult):
    """姿态估计结果"""
    st.markdown("**关键点**")

    if test_media['type'] == 'image':
        image = test_media['original'].copy()

        for i, kp in enumerate(result.keypoints):
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(image, (int(x), int(y)), 8, (0, 255, 0), -1)

        st.image(image, use_column_width=True)

    if result.keypoints:
        st.info(f"{len(result.keypoints)} 个关键点")
    else:
        st.warning("未检测到关键点")


def render_face_landmark_result(test_media: dict, result: InferenceResult):
    """人脸关键点结果"""
    if test_media['type'] == 'image':
        image = test_media['original'].copy()
        for kp in result.keypoints:
            x, y = kp[0], kp[1]
            cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        st.image(image, use_column_width=True)

    if result.keypoints:
        st.info(f"{len(result.keypoints)} 个关键点")
    else:
        st.warning("未检测到人脸")


def render_audio_result(result: InferenceResult):
    """音频分类结果"""
    st.markdown("**分类结果**")

    if result.classifications:
        for label, score in result.classifications.items():
            st.write(f"{label}: {score:.2%}")
            st.progress(score)

        top_class = max(result.classifications.items(), key=lambda x: x[1])
        st.success(f"最可能: {top_class[0]} ({top_class[1]:.2%})")
    else:
        st.warning("无法识别")


def render_classification_result(result: InferenceResult):
    """图像分类结果"""
    if result.classifications:
        top_class = result.classifications.get('top_class', '未知')
        confidence = result.classifications.get('confidence', 0)
        st.metric("预测类别", f"类别 {top_class}", f"{confidence:.2%}")
    else:
        st.warning("无法分类")


# 页面内容（函数定义之后）
st.title("模型验证")

col_select, col_test = st.columns([1, 2])

with col_select:
    st.subheader("选择模型")
    selected_model = render_model_selector()

with col_test:
    if selected_model:
        st.subheader("测试推理")
        render_test_section(selected_model)
    else:
        st.info("请先选择一个模型")
"""
模型管理页面 - 上传、查看、删除模型
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.model_registry import ModelRegistry, ModelType, ModelFormat, ModelMetadata

# 页面配置
st.set_page_config(
    page_title="模型管理 - 宝宝监护器",
    layout="wide"
)


@st.cache_resource
def get_registry():
    return ModelRegistry()



def main():
    st.title("模型管理")

    col_upload, col_list = st.columns([1, 2])

    with col_upload:
        st.subheader("📤 上传模型")
        render_upload_section()

    with col_list:
        st.subheader("📋 已上传模型")
        render_model_list()


def render_upload_section():
    """上传区域"""

    model_type = st.selectbox(
        "模型类型",
        options=[
            ModelType.DETECTION,
            ModelType.POSE,
            ModelType.AUDIO,
            ModelType.FACE_LANDMARK,
            ModelType.CLASSIFICATION,
        ],
        format_func=lambda x: x.value
    )

    uploaded_file = st.file_uploader(
        "选择模型文件",
        type=['pt', 'onnx', 'engine', 'pth', 'tflite']
    )

    if uploaded_file is not None:
        st.info(f"文件: {uploaded_file.name} ({uploaded_file.size/(1024*1024):.2f} MB)")

        model_name = st.text_input(
            "模型名称",
            value=Path(uploaded_file.name).stem
        )

        description = st.text_area(
            "描述",
            placeholder="模型用途说明"
        )

        if st.button("上传", type="primary"):
            success = save_model(uploaded_file, model_name, model_type, description)
            if success:
                st.success(f"✅ '{model_name}' 上传成功")
                st.rerun()


def save_model(uploaded_file, model_name: str, model_type: ModelType, description: str) -> bool:
    """保存模型文件"""
    registry = get_registry()

    model_dir = Path("models") / model_type.name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    model_id = f"{model_type.name.lower()}_{int(datetime.now().timestamp())}"

    suffix = Path(uploaded_file.name).suffix.lower()
    model_format = {
        '.pt': ModelFormat.PYTORCH,
        '.pth': ModelFormat.PYTORCH,
        '.onnx': ModelFormat.ONNX,
        '.engine': ModelFormat.TENSORRT,
        '.plan': ModelFormat.TENSORRT,
        '.tflite': ModelFormat.TFLITE,
    }.get(suffix, ModelFormat.PYTORCH)

    file_path = model_dir / f"{model_id}{suffix}"
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    metadata = ModelMetadata(
        model_id=model_id,
        name=model_name,
        file_path=str(file_path),
        model_type=model_type,
        model_format=model_format,
        file_size=uploaded_file.size,
        upload_time=datetime.now(),
        description=description
    )

    registry.register(metadata)
    return True


def render_model_list():
    """模型列表"""
    registry = get_registry()
    models = registry.list_all()

    if not models:
        st.info("暂无模型，请先上传")
        return

    # 按类型分组
    models_by_type = {}
    for model in models:
        if model.model_type not in models_by_type:
            models_by_type[model.model_type] = []
        models_by_type[model.model_type].append(model)

    for model_type, type_models in models_by_type.items():
        with st.expander(f"{model_type.value} ({len(type_models)})"):
            for model in type_models:
                render_model_card(model)


def render_model_card(model: ModelMetadata):
    """单个模型卡片"""
    col_name, col_info, col_btn = st.columns([2, 2, 1])

    with col_name:
        st.markdown(f"**{model.name}**")
        st.caption(f"ID: `{model.model_id}`")

    with col_info:
        st.caption(f"{model.model_format.value.split()[0]} | {model.file_size/(1024*1024):.1f} MB")

    with col_btn:
        if st.button("删除", key=f"del_{model.model_id}"):
            delete_model(model)
            st.rerun()

    if model.description:
        st.text(model.description)

    st.divider()


def delete_model(model: ModelMetadata) -> bool:
    """删除模型"""
    registry = get_registry()
    file_path = Path(model.file_path)
    if file_path.exists():
        file_path.unlink()
    return registry.unregister(model.model_id)


if __name__ == "__main__":
    main()
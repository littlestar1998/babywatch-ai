"""
设备状态页面 - 展示设备与 GPU 状态
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_registry import ModelRegistry
from core.gpu_monitor import GPUMonitor


@st.cache_resource
def get_registry():
    return ModelRegistry()


def render_device_and_gpu_status():
    gpu_info = GPUMonitor.get_info()
    jetson_info = GPUMonitor.get_jetson_info()

    st.subheader("设备与 GPU 状态")

    col1, col2 = st.columns(2)

    with col1:
        if jetson_info.get("is_jetson"):
            st.success("设备: {}".format(jetson_info.get('gpu_type', 'Jetson')))
        else:
            st.warning("设备: 非 Jetson")

        if "cuda_device_name" in jetson_info:
            st.caption("GPU: {}".format(jetson_info['cuda_device_name']))
        st.caption("架构: {}".format(jetson_info.get('architecture', '--')))

    with col2:
        if "cuda_compute_capability" in jetson_info:
            st.caption("计算能力: {}".format(jetson_info['cuda_compute_capability']))
        if jetson_info.get("tegra_version") and jetson_info.get("tegra_version") != "Unknown":
            st.caption("Tegra: {}".format(jetson_info['tegra_version']))

    st.markdown("---")

    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric("显存使用", "{} MB".format(int(gpu_info.memory_used_mb)))
    with metric2:
        st.metric("总显存", "{} MB".format(int(gpu_info.memory_total_mb)) if gpu_info.memory_total_mb else "--")
    with metric3:
        st.metric("GPU利用率", "{}%".format(int(gpu_info.gpu_utilization)))
    with metric4:
        st.metric("温度", "{} C".format(int(gpu_info.temperature)) if gpu_info.temperature else "--")

    if gpu_info.memory_total_mb > 0:
        st.markdown("**显存占用**")
        st.progress(min(gpu_info.memory_percent, 100.0) / 100.0)
        st.caption("当前占用 {:.1f}%".format(gpu_info.memory_percent))

    if gpu_info.power_draw is not None:
        st.caption("功耗: {:.1f} W".format(gpu_info.power_draw))

    st.markdown("---")
    if st.button("清理 GPU 缓存", type="primary"):
        GPUMonitor.clear_gpu_cache()
        st.success("GPU 缓存已清理")
        st.rerun()


def render_model_monitor_panel():
    registry = get_registry()
    models = registry.list_all()

    st.subheader("模型信息")
    st.metric("模型总数", registry.count())

    counts = registry.count_by_type()
    if counts:
        st.markdown("**各类型数量**")
        for type_name, count in counts.items():
            st.caption("{}: {} 个".format(type_name, count))
    else:
        st.info("暂无模型")

    st.markdown("---")

    if models:
        total_size = sum(m.file_size for m in models) / (1024 * 1024)
        st.metric("模型总占用", "{:.1f} MB".format(total_size))

        latest_model = max(models, key=lambda m: m.upload_time)
        st.markdown("**最近上传**")
        st.caption("名称: {}".format(latest_model.name))
        st.caption("时间: {}".format(latest_model.upload_time.strftime('%Y-%m-%d %H:%M:%S')))
        st.caption("格式: {}".format(latest_model.model_format.value))
    else:
        st.caption("最近上传: --")


# 页面内容（函数定义之后）
st.title("设备状态")

col_left, col_right = st.columns([3, 2])

with col_left:
    render_device_and_gpu_status()

with col_right:
    render_model_monitor_panel()
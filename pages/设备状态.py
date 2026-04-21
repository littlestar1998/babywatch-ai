"""
设备状态页面 - 展示设备与 GPU 状态
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.model_registry import ModelRegistry
from core.gpu_monitor import GPUMonitor

# 页面配置
st.set_page_config(
    page_title="设备状态 - 宝宝监护器",
    layout="wide"
)


@st.cache_resource
def get_registry():
    return ModelRegistry()


def main():
    st.title("设备状态")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_device_and_gpu_status()

    with col_right:
        render_model_monitor_panel()


def render_device_and_gpu_status():
    gpu_info = GPUMonitor.get_info()
    jetson_info = GPUMonitor.get_jetson_info()

    st.subheader("设备与 GPU 状态")

    col1, col2 = st.columns(2)

    with col1:
        if jetson_info.get("is_jetson"):
            st.success(f"设备: {jetson_info.get('gpu_type', 'Jetson')}")
        else:
            st.warning("设备: 非 Jetson")

        if "cuda_device_name" in jetson_info:
            st.caption(f"GPU: {jetson_info['cuda_device_name']}")
        st.caption(f"架构: {jetson_info.get('architecture', '--')}")

    with col2:
        if "cuda_compute_capability" in jetson_info:
            st.caption(f"计算能力: {jetson_info['cuda_compute_capability']}")
        if jetson_info.get("tegra_version") and jetson_info.get("tegra_version") != "Unknown":
            st.caption(f"Tegra: {jetson_info['tegra_version']}")

    st.markdown("---")

    metric1, metric2, metric3, metric4 = st.columns(4)
    with metric1:
        st.metric("显存使用", f"{gpu_info.memory_used_mb:.0f} MB")
    with metric2:
        st.metric("总显存", f"{gpu_info.memory_total_mb:.0f} MB" if gpu_info.memory_total_mb else "--")
    with metric3:
        st.metric("GPU利用率", f"{gpu_info.gpu_utilization:.0f}%")
    with metric4:
        st.metric("温度", f"{gpu_info.temperature:.0f}°C" if gpu_info.temperature else "--")

    if gpu_info.memory_total_mb > 0:
        st.markdown("**显存占用**")
        st.progress(min(gpu_info.memory_percent, 100.0) / 100.0)
        st.caption(f"当前占用 {gpu_info.memory_percent:.1f}%")

    if gpu_info.power_draw is not None:
        st.caption(f"功耗: {gpu_info.power_draw:.1f} W")

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
            st.caption(f"{type_name}: {count} 个")
    else:
        st.info("暂无模型")

    st.markdown("---")

    if models:
        total_size = sum(m.file_size for m in models) / (1024 * 1024)
        st.metric("模型总占用", f"{total_size:.1f} MB")

        latest_model = max(models, key=lambda m: m.upload_time)
        st.markdown("**最近上传**")
        st.caption(f"名称: {latest_model.name}")
        st.caption(f"时间: {latest_model.upload_time.strftime('%Y-%m-%d %H:%M:%S')}")
        st.caption(f"格式: {latest_model.model_format.value}")
    else:
        st.caption("最近上传: --")

    

if __name__ == "__main__":
    main()

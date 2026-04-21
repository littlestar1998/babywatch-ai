"""
宝宝监护器 - 主页面
"""

import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.model_registry import ModelRegistry

# 页面配置
st.set_page_config(
    page_title="宝宝监护器",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_registry():
    return ModelRegistry()


def main():
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.title("👶 宝宝监护器")
        st.markdown("""
        ### 智能 AI 守护宝宝安全

        运行在 **Jetson Orin Nano** 上的智能宝宝监护系统，
        利用 AI 技术实时监测宝宝状态，提供全方位的安全守护。
        """)

        st.markdown("---")
        render_features_overview()

    with col_side:
        render_home_side_panel()


def render_home_side_panel():
    registry = get_registry()
    models = registry.list_all()

    st.markdown("### 📌 概览")
    st.metric("模型数量", registry.count())

    if models:
        counts = registry.count_by_type()
        for type_name, count in counts.items():
            st.caption(f"{type_name}: {count} 个")

        total_size = sum(m.file_size for m in models) / (1024 * 1024)
        st.caption(f"存储占用: {total_size:.1f} MB")
    else:
        st.info("暂无模型")

    st.markdown("---")
    st.caption("使用侧边栏导航到各功能页面")


def render_features_overview():
    st.markdown("### 🎯 核心功能")

    features = [
        {"icon": "😴", "name": "睡眠检测", "desc": "监测宝宝睡眠状态"},
        {"icon": "⚠️", "name": "口鼻遮挡检测", "desc": "防止窒息风险"},
        {"icon": "🔊", "name": "哭声检测", "desc": "识别宝宝哭声"},
        {"icon": "🚧", "name": "虚拟围栏", "desc": "设置安全区域"},
        {"icon": "📸", "name": "精彩抓拍", "desc": "捕捉可爱瞬间"},
        {"icon": "🏃", "name": "动作识别", "desc": "识别翻身、爬行等"},
        {"icon": "🤔", "name": "意图识别", "desc": "分析宝宝意图"},
    ]

    cols = st.columns(4)
    for i, feature in enumerate(features):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; text-align: center; min-height: 140px;">
                    <div style="font-size: 30px;">{feature['icon']}</div>
                    <div style="font-weight: bold; margin-top: 10px;">{feature['name']}</div>
                    <div style="color: #666; font-size: 12px; margin-top: 6px;">{feature['desc']}</div>
                    <div style="color: #FFA500; font-size: 11px; margin-top: 10px;">开发中</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

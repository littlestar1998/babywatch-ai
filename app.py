"""
宝宝监护器 - 主页面
"""

import streamlit as st


def render_home():
    """主页内容"""
    st.title("宝宝监护器")
    st.markdown("""
    ### 智能 AI 守护宝宝安全

    运行在 **Jetson Orin Nano** 上的智能宝宝监护系统，
    利用 AI 技术实时监测宝宝状态，提供全方位的安全守护。
    """)

    st.markdown("---")
    render_features_overview()


def render_features_overview():
    st.markdown("### 核心功能")

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


# 页面配置
st.set_page_config(
    page_title="宝宝监护器",
    page_icon=":baby:",
    layout="wide",
)

# 定义页面结构（二级菜单）
home_page = st.Page(render_home, title="首页", default=True)

pages = {
    "": [home_page],  # 首页放在顶层
    "模型": [
        st.Page("pages/模型管理.py", title="模型管理"),
        st.Page("pages/模型验证.py", title="模型验证"),
    ],
    "设备": [
        st.Page("pages/设备状态.py", title="设备状态"),
    ],
}

# 导航
page = st.navigation(pages, position="sidebar", expanded=True)
page.run()
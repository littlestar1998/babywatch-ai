# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

这是一个**宝宝监护器**项目，使用 Python 开发，运行在 Jetson Orin Nano 上（带 GPU，可运行 AI 算法）。后台使用 Streamlit 构建。

## 核心功能

1. **睡眠检测** - 检测宝宝的睡眠状态
2. **面部口鼻遮挡检测** - 检测到遮挡时触发报警，防止窒息风险
3. **哭声检测** - 音频识别宝宝哭声
4. **虚拟围栏** - 设置安全区域，宝宝靠近围栏时提醒
5. **精彩抓拍** - 自动捕捉宝宝可爱瞬间
6. **动作识别** - 识别宝宝在做什么（如翻身、爬行等）
7. **意图识别** - 分析宝宝想要做什么（如想要抱抱、想要玩具等）

## 运行应用

```bash
# 启动 Streamlit 应用
streamlit run app.py

# 或使用 uv
uv run streamlit run app.py
```

## 依赖管理

使用 **uv** 管理 Python 依赖：

```bash
uv add <package>    # 添加依赖
uv sync             # 同步依赖
uv lock             # 锁定依赖版本
```

## 硬件环境

- Jetson Orin Nano（带 GPU）
- `/dev/video0` - RGB 摄像头
- `/dev/video1` - 深度摄像头（奥比中光）

## 给 AI 小白的说明

本项目涉及多种 AI 技术。在实现时，我会解释：

- **模型选择**：为什么选择某个模型，它的优缺点
- **数据流**：数据从输入到输出的完整路径
- **推理过程**：模型如何处理数据并给出结果
- **Jetson 优化**：如何利用 GPU 加速，TensorRT 优化等

遇到 AI 相关代码时，会添加详细注释解释原理和逻辑。

## 语言要求

所有对话和代码注释均使用中文。
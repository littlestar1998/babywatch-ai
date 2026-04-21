"""
GPU 监控器 - 实时监控 GPU 使用情况

【AI 初学者说明】
在 Jetson 设备上，GPU 资源有限，需要监控：
- 显存使用量：模型和数据占用的内存
- GPU 利用率：GPU 计算单元的使用程度
- 温度：防止过热降频

这些信息帮助判断模型是否适合在设备上运行。
Jetson 使用"统一内存"架构，CPU 和 GPU 共享同一块物理内存，
所以内存管理比传统 GPU 更复杂。
"""

import subprocess
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import platform


@dataclass
class GPUInfo:
    """
    GPU 状态信息数据类

    【AI 初学者说明】
    这个数据类存储 GPU 的实时状态：
    - memory_used_mb: 已使用的显存（MB）
    - memory_total_mb: 总显存（MB）
    - memory_percent: 显存使用百分比
    - gpu_utilization: GPU 计算单元利用率
    - temperature: GPU 温度（摄氏度）
    - power_draw: 当前功耗（瓦特）
    """
    memory_used_mb: float          # 已用显存
    memory_total_mb: float         # 总显存
    memory_percent: float          # 显存使用百分比
    gpu_utilization: float        # GPU 利用率
    temperature: Optional[float]   # 温度（摄氏度）
    power_draw: Optional[float]    # 功耗（瓦特）

    def is_memory_warning(self, threshold: float = 80.0) -> bool:
        """检查显存是否超过警告阈值"""
        return self.memory_percent > threshold

    def is_temperature_warning(self, threshold: float = 85.0) -> bool:
        """检查温度是否超过警告阈值"""
        if self.temperature is None:
            return False
        return self.temperature > threshold


class GPUMonitor:
    """
    Jetson GPU 监控器

    【AI 初学者说明】
    Jetson 设备有特殊的监控工具：
    - tegrastats: 系统级监控工具，显示 RAM、GPU、CPU、温度等
    - jtop: Jetson 专用监控工具（需要安装 jetson-stats）

    这个类封装了这些工具，提供统一的监控接口。
    """

    @staticmethod
    def get_info() -> GPUInfo:
        """
        获取 GPU 状态

        【AI 初学者说明】
        这个方法读取系统状态并返回 GPUInfo 对象。
        会尝试多种方式获取数据：
        1. 使用 tegrastats（Jetson 原生工具）
        2. 使用 jtop（Jetson-stats 工具）
        3. 使用 PyTorch CUDA API（通用方法）
        """
        # 首先尝试 tegrastats（Jetson 专用）
        info = GPUMonitor._get_tegrastats_info()
        if info:
            return info

        # 其次尝试 PyTorch CUDA
        info = GPUMonitor._get_cuda_info()
        if info:
            return info

        # 最后返回默认值
        return GPUInfo(
            memory_used_mb=0,
            memory_total_mb=0,
            memory_percent=0,
            gpu_utilization=0,
            temperature=None,
            power_draw=None
        )

    @staticmethod
    def _get_tegrastats_info() -> Optional[GPUInfo]:
        """
        使用 tegrastats 获取 Jetson GPU 信息

        【AI 初学者说明】
        tegrastats 是 Jetson 设备内置的系统监控工具，
        输出格式示例：
        RAM 4000/8000MB (lfb 3000MB), GPU 1000/4000MB (lfb 500MB), CPU [20%@1024]
        TEMPERATURE: GPU 45C, CPU 50C
        POWER: GPU 5W, CPU 2W

        我们解析这个输出，提取 GPU 相关数据。
        """
        try:
            # 执行 tegrastats 命令
            result = subprocess.run(
                ['tegrastats'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return None

            output = result.stdout.strip()
            return GPUMonitor._parse_tegrastats(output)

        except subprocess.TimeoutExpired:
            return None
        except FileNotFoundError:
            # tegrastats 不存在（可能不是 Jetson 设备）
            return None
        except Exception as e:
            print(f"[GPU监控] tegrastats 解析失败: {e}")
            return None

    @staticmethod
    def _parse_tegrastats(output: str) -> Optional[GPUInfo]:
        """
        解析 tegrastats 输出

        【AI 初学者说明】
        tegrastats 输出是一行复杂的文本，需要用正则表达式提取数据。

        示例输出：
        RAM 4015/7847MB (lfb 4x2048MB), GPU 0/7847MB (lfb 2x4MB), EMC 4@2048MHz...
        """
        try:
            # 解析 GPU 内存
            # 格式: GPU 1000/4000MB
            gpu_mem_match = re.search(r'GPU\s+(\d+)/(\d+)MB', output)
            if gpu_mem_match:
                memory_used = float(gpu_mem_match.group(1))
                memory_total = float(gpu_mem_match.group(2))
            else:
                # 尝试另一种格式（Jetson Orin）
                # 格式: GPU 1000/4000 (no unit)
                gpu_mem_match = re.search(r'GPU\s+(\d+)/(\d+)', output)
                if gpu_mem_match:
                    memory_used = float(gpu_mem_match.group(1))
                    memory_total = float(gpu_mem_match.group(2))
                else:
                    memory_used = 0
                    memory_total = 0

            memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

            # 解析 GPU 利用率
            # 格式: GPU [30%@500] 或 GPU 30%
            gpu_util_match = re.search(r'GPU\s+\[(\d+)%', output)
            if gpu_util_match:
                gpu_utilization = float(gpu_util_match.group(1))
            else:
                gpu_util_match = re.search(r'GPU\s+(\d+)%', output)
                if gpu_util_match:
                    gpu_utilization = float(gpu_util_match.group(1))
                else:
                    gpu_utilization = 0

            # 解析温度
            # 格式: GPU@45C 或 temperature GPU 45C
            temp_match = re.search(r'GPU@(\d+)C', output, re.IGNORECASE)
            if temp_match:
                temperature = float(temp_match.group(1))
            else:
                temp_match = re.search(r'temperature.*?GPU\s+(\d+)C', output, re.IGNORECASE)
                temperature = float(temp_match.group(1)) if temp_match else None

            # 解析功耗
            # 格式: POM_5V_IN 5000/5000
            power_match = re.search(r'POM_5V_GPU_IN\s+(\d+)/(\d+)', output)
            power_draw = float(power_match.group(1)) / 1000 if power_match else None  # 转换为瓦特

            return GPUInfo(
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                memory_percent=memory_percent,
                gpu_utilization=gpu_utilization,
                temperature=temperature,
                power_draw=power_draw
            )

        except Exception as e:
            print(f"[GPU监控] tegrastats 解析错误: {e}")
            return None

    @staticmethod
    def _get_cuda_info() -> Optional[GPUInfo]:
        """
        使用 PyTorch CUDA API 获取 GPU 信息

        【AI 初学者说明】
        如果 tegrastats 不可用，可以使用 PyTorch 的 CUDA 接口：
        - torch.cuda.memory_allocated(): 已分配的显存
        - torch.cuda.memory_reserved(): 已预留的显存
        - torch.cuda.get_device_properties(): 设备属性

        这种方法更通用，但信息不如 tegrastats 全面。
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return None

            # 获取设备属性
            device = torch.device('cuda')
            props = torch.cuda.get_device_properties(device)

            # 获取内存信息
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            memory_reserved = torch.cuda.memory_reserved() / (1024 * 1024)   # MB
            memory_total = props.total_memory / (1024 * 1024)  # MB

            memory_percent = (memory_allocated / memory_total * 100) if memory_total > 0 else 0

            return GPUInfo(
                memory_used_mb=memory_allocated,
                memory_total_mb=memory_total,
                memory_percent=memory_percent,
                gpu_utilization=0,  # PyTorch 不提供利用率
                temperature=None,   # PyTorch 不提供温度
                power_draw=None     # PyTorch 不提供功耗
            )

        except ImportError:
            return None
        except Exception as e:
            print(f"[GPU监控] CUDA 信息获取失败: {e}")
            return None

    @staticmethod
    def get_jetson_info() -> Dict[str, Any]:
        """
        获取 Jetson 设备信息

        【AI 初学者说明】
        了解设备型号有助于选择合适的模型：
        - Jetson Nano: 适合轻量模型（如 YOLOv5s）
        - Jetson Orin Nano: 中等规模模型（如 YOLOv8m）
        - Jetson AGX Orin: 大型模型（如 YOLOv8x）

        返回：
        - tegra_version: Tegra 系统版本
        - gpu_name: GPU 名称
        - architecture: 系统架构（通常为 aarch64）
        - memory_total: 总内存（MB）
        """
        info = {
            'tegra_version': 'Unknown',
            'gpu_name': 'Unknown',
            'architecture': platform.machine(),
            'memory_total': 0,
            'is_jetson': False
        }

        # 检测是否是 Jetson 设备
        tegra_release_path = Path('/etc/nv_tegra_release')
        if tegra_release_path.exists():
            info['is_jetson'] = True

            try:
                with open(tegra_release_path, 'r') as f:
                    version = f.read().strip()
                    info['tegra_version'] = version
            except:
                pass

            # 尝试读取 Jetson 型号
            try:
                import subprocess
                result = subprocess.run(
                    ['cat', '/proc/device-tree/model'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    model = result.stdout.strip()
                    info['gpu_name'] = model

                    # 从型号推断内存大小
                    if 'Orin Nano' in model:
                        info['memory_total'] = 8192  # 8GB
                        info['gpu_type'] = 'Orin Nano'
                    elif 'Orin NX' in model:
                        info['memory_total'] = 16384  # 16GB
                        info['gpu_type'] = 'Orin NX'
                    elif 'AGX Orin' in model:
                        info['memory_total'] = 32768  # 32GB
                        info['gpu_type'] = 'AGX Orin'
                    elif 'Nano' in model:
                        info['memory_total'] = 4096  # 4GB
                        info['gpu_type'] = 'Nano'

            except:
                pass

        # 使用 PyTorch 获取 GPU 信息作为补充
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info['cuda_device_name'] = props.name
                info['cuda_memory_total'] = props.total_memory / (1024 * 1024)  # MB
                info['cuda_compute_capability'] = f"{props.major}.{props.minor}"
        except:
            pass

        return info

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """
        获取内存详细信息

        【AI 初学者说明】
        Jetson 使用统一内存，需要同时关注：
        - 系统总内存
        - GPU 使用的内存
        - 可用内存

        返回各种内存指标的字典。
        """
        memory_info = {}

        # 使用 PyTorch CUDA
        try:
            import torch
            if torch.cuda.is_available():
                memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                memory_info['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 * 1024)

                props = torch.cuda.get_device_properties(0)
                memory_info['cuda_total_mb'] = props.total_memory / (1024 * 1024)
        except:
            pass

        # 使用系统命令
        try:
            # 读取 /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        memory_info['system_total_kb'] = float(re.search(r'\d+', line).group())
                    elif 'MemFree' in line:
                        memory_info['system_free_kb'] = float(re.search(r'\d+', line).group())
                    elif 'MemAvailable' in line:
                        memory_info['system_available_kb'] = float(re.search(r'\d+', line).group())
        except:
            pass

        return memory_info

    @staticmethod
    def clear_gpu_cache() -> None:
        """
        清理 GPU 缓存

        【AI 初学者说明】
        当切换模型或完成推理后，应该清理 GPU 缓存：
        1. 清理 Python 垃圾回收
        2. 清理 PyTorch CUDA 缓存
        3. 同步 GPU 状态

        这可以释放未使用的显存，为新模型腾出空间。
        """
        import gc

        # Python 垃圾回收
        gc.collect()

        # PyTorch CUDA 缓存清理
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

    @staticmethod
    def format_info_display(info: GPUInfo) -> str:
        """
        格式化 GPU 信息用于显示

        【AI 初学者说明】
        将 GPUInfo 对象转换为易读的字符串，
        用于在 UI 中显示。
        """
        lines = [
            f"显存: {info.memory_used_mb:.0f} / {info.memory_total_mb:.0f} MB ({info.memory_percent:.1f}%)",
            f"GPU 利用率: {info.gpu_utilization:.1f}%",
        ]

        if info.temperature is not None:
            lines.append(f"温度: {info.temperature:.1f}°C")

        if info.power_draw is not None:
            lines.append(f"功耗: {info.power_draw:.1f} W")

        return "\n".join(lines)
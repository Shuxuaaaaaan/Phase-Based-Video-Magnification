# Phase-Based Video Magnification

```
██████╗ ██████╗ ███╗   ███╗ ██████╗ ███╗   ███╗ █████╗ 
██╔══██╗██╔══██╗████╗ ████║██╔═══██╗████╗ ████║██╔══██╗
██████╔╝██████╔╝██╔████╔██║██║   ██║██╔████╔██║███████║
██╔═══╝ ██╔══██╗██║╚██╔╝██║██║   ██║██║╚██╔╝██║██╔══██║
██║     ██████╔╝██║ ╚═╝ ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║
╚═╝     ╚═════╝ ╚═╝     ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝
```

Phase Based Video Motion Processing for color and motion magnification.  
This is a robust Python 3 implementation based on the original ACM SIGGRAPH 2013 paper.  
This project aims to support fast Python execution through multi-processing and GPU acceleration.

基于相位视频运动处理的颜色与运动放大工具。  
这是基于2013年 ACM SIGGRAPH 论文的一个稳健的Python 3版本实现。  
本项目致力于通过多进程与 GPU 加速提供更快捷的处理管线。

> [!NOTE]  
> This project is a modern refactor of the original Python code implemented during the Lorentz Center workshop (ICT with Industry: motion microscope, 2015).  
> 本项目是根据2015年 Lorentz Center motion microscope 研讨会上的原版Python代码进行的现代化重构与架构升级。

## Requirements / 环境要求

- Python >= 3.9
- [UV](https://docs.astral.sh/uv/) — Fast Python Package manager / 高速包管理器
- OpenCV, Numpy, Scipy dependencies

## Installation / 安装

```bash
# Setup UV environment and install dependencies / 初始化环境并安装依赖
uv sync
```

## Usage / 使用方法

```bash
uv run python src/evm_phase.py -v <video_path> -s <saving_path> [options]
```

### Arguments / 参数

| Argument / 参数 | Short / 缩写 | Description / 说明 | Default / 默认值 |
|----------|-------|-------------|---------|
| `--video_path` | `-v` | Input video path / 输入视频路径 | *required / 必填* |
| `--saving_path` | `-s` | Output sequence path / 输出序列路径 `.mp4`/`.avi` | *required / 必填* |
| `--alpha` | `-a` | Amplification factor / 放大系数 | `20` |
| `--low_omega` | `-lo` | Min frequency bandwidth / 最低滤波频率 | `72` |
| `--high_omega` | `-ho` | Max frequency bandwidth / 最高滤波频率 | `92` |
| `--window_size` | `-ws` | Ideal filter sliding window / 理想滤波滑动窗口 | `30` |
| `--max_frames` | `-mf` | Max frames to process / 最大处理帧数限制 | `60000` |
| `--fps` | `-f` | FPS for bandpass (-1 for native) / 带通滤波帧率参考 | `600` |

### Examples / 示例

```bash
# Magnify string vibrations in a guitar video / 放大吉他拨弦的微波运动
uv run python src/evm_phase.py -v data/resources/guitar.mp4 -s data/results/guitar.mp4 -a 20 -lo 72 -ho 92
```

## References / 参考文献

- [Phase Based Video Motion Processing (ACM SIGGRAPH 2013)](http://people.csail.mit.edu/nwadhwa/phase-video/)
  > Authors: Neal Wadhwa, Michael Rubinstein, Frédo Durand, William T. Freeman

- [Steerable-filter Phase Decomposer (andreydung)](https://github.com/andreydung/Steerable-filter)

## Credits & Authors / 原作者与致谢

- Original Python 2.7 Implementation (2015 Lorentz Center workshop): Joao Bastos, Elsbeth van Dam, Coert van Gemeren, Jan van Gemert, Amogh Gudi, Julian Kooij, Malte Lorbach, Claudio Martella, Ronald Poppe.
- Modern Python 3 Refactor & Acceleration Architecture by Shuxuan

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

## Docker Deployment / 容器化部署

The project includes a multi-architecture `Dockerfile`, a `docker-compose.yml`, and an automated PowerShell script to build container images for both AMD64 (Desktop/Server) and ARM64 (Raspberry Pi/Mac M-Series). CuPy dependency is dynamically skipped on ARM64 builds.
本项目内置了多架构 `Dockerfile`，`docker-compose.yml` 以及一键构建脚本，用以双平台输出 AMD64 与 ARM64 的静态镜像。CuPy (CUDA) 依赖在 ARM64 平台上会被自动安全跳过。

**1. Create Images / 跨平台构建镜像:**

```powershell
# Requires Docker Desktop running buildx / 需要开启 Docker Desktop
./build_docker_images.ps1
```
This will cleanly cross-compile and generate two static images: `docker_images/evm-phase-app_amd64.tar` and `evm-phase-app_arm64.tar`.

**2. Deploy with Docker Compose / 部署与执行:**

Transfer the `.tar` image and the `docker-compose.yml` to your target device, then load and run:
将在步骤一生成的镜像压缩包和根目录的 `docker-compose.yml` 传输到您的目标设备（如树莓派）上。

```bash
# Example for ARM64 (Raspberry Pi / Mac):
# 1. Load the archive into docker / 载入本地镜像
docker load -i docker_images/evm-phase-app_arm64.tar

# 2. Tag the loaded image so docker-compose recognizes it / 为镜像打上本地标签以匹配 compose 配置
docker tag evm-phase-app:local-arm64 evm-phase-app:local

# 3. Run the application logic via Compose / 使用 compose 运行放大算法
docker compose run --rm evm-app -v data/resources/guitar.mp4 -s data/results/out.mp4 -a 20 -lo 72 -ho 92 -t 1 -acc cpu
```
> [!NOTE]
> When using Docker Compose, the `./data` directory from your host machine is automatically mounted to `/app/data` inside the container. Place your videos in `data/resources` and retrieve results from `data/results`. 
> > If you want to use Nvidia CUDA acceleration on an AMD64 Linux machine, ensure the `nvidia-container-toolkit` is installed, and uncomment the `deploy.resources` block inside the `docker-compose.yml` file.

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
| `--threads` | `-t` | Number of CPU workers / 多线程并发核心数 | `1` |
| `--accel` | `-acc` | Acceleration backend / 加速后端选项 `cpu`/`cuda` | `cpu` |

### Examples / 示例

```bash
# Standard Single Thread CPU / 标准单线程 (~3.02 FPS)
uv run python src/evm_phase.py -v data/resources/guitar.mp4 -s data/results/guitar.mp4 -a 20 -lo 72 -ho 92 -t 1 -acc cpu

# 8x CPU Multithreading / 八线程加速 (Speedup ~2.66x)
uv run python src/evm_phase.py -v data/resources/guitar.mp4 -s data/results/guitar.mp4 -a 20 -lo 72 -ho 92 -t 8 -acc cpu

# CUDA Hardware Acceleration / GPU 加速 (Speedup ~4x to 26x depending on video length)
uv run python src/evm_phase.py -v data/resources/guitar.mp4 -s data/results/guitar.mp4 -a 20 -lo 72 -ho 92 -t 1 -acc cuda
```

### Performance & Limitations / 性能与局限说明

- **CPU Multithreading (`-t > 1`)**: [DEPRECATION WARNING] EVM processing involves huge complex NDArrays (typically 200MB+ per frame) constructed by the Steerable Pyramid filterbank. Due to Python's GIL and strict pickling constraints, deep parallel processing incurs heavy IPC/serialization overhead. Multi-threading is currently known to deadlock and hang during the temporal filtering phase on long clips. DO NOT run with threads > 1.
- **CUDA Acceleration (`-acc cuda`)**: Leveraging Nvidia GPUs strictly avoids cross-process memory swapping by transferring computations directly to High-Bandwidth VRAM via `cupy` backend. Due to the CUDA Context initialization latency (about 1-2 seconds), performance gains manifest best on sequences longer than 5 seconds. Short 15-frame scripts might only show 2.5x speedups, while long videos witness 20x+ acceleration.

- **CPU 多线程 (`-t > 1`)**：[弃用警告] EVM处理涉及由可控金字塔滤波器组构建的庞大而复杂的NDArray（通常每帧200MB以上）。由于Python的GIL和严格的序列化限制，深度并行处理会产生灾难性的IPC拥堵。当前架构下的多线程处理在长视频序列化尾盘存在已知的队列死锁与崩溃问题，不推荐使用！请始终保持 `-t 1`。

- **CUDA 加速 (`-acc cuda`)**：利用Nvidia GPU，通过`cupy`后端将计算直接传输到高带宽显存，从而严格避免跨进程内存交换。由于CUDA上下文初始化延迟（约1-2秒），性能提升在超过5秒的序列上最为明显。较短的15帧脚本可能仅显示2.5倍的加速，而长视频则可实现20倍以上的加速。

## References / 参考文献

- [Phase Based Video Motion Processing (ACM SIGGRAPH 2013)](http://people.csail.mit.edu/nwadhwa/phase-video/)
  > Authors: Neal Wadhwa, Michael Rubinstein, Frédo Durand, William T. Freeman

- [Steerable-filter Phase Decomposer (andreydung)](https://github.com/andreydung/Steerable-filter)

## Credits & Authors / 原作者与致谢

- Original Python 2.7 Implementation (2015 Lorentz Center workshop): Joao Bastos, Elsbeth van Dam, Coert van Gemeren, Jan van Gemert, Amogh Gudi, Julian Kooij, Malte Lorbach, Claudio Martella, Ronald Poppe.
- Modern Python 3 Refactor & Acceleration Architecture by Shuxuan

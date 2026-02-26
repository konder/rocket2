# ROCKET-2 macOS ARM64 (Apple Silicon) 部署指南

本文档记录在 Apple Silicon Mac (M1/M2/M3/M4) 上原生部署 ROCKET-2 的完整步骤。

## 前置条件

- macOS 14+ (Sonoma) Apple Silicon Mac
- [Miniforge](https://github.com/conda-forge/miniforge) (ARM64 版)
- [Homebrew](https://brew.sh/)
- Git

## 1. 安装系统依赖

### 1.1 安装 x86_64 JDK 8 (Temurin)

MineStudio 的 Minecraft 模拟器引擎需要 JDK 8，且引擎打包的 LWJGL 原生库是 x86_64 架构，
必须使用 x86_64 JDK 通过 Rosetta 2 运行。

```bash
# 安装 Rosetta 2（如果尚未安装）
softwareupdate --install-rosetta --agree-to-license

# 通过 Homebrew 安装 x86_64 版 Temurin JDK 8
arch -x86_64 brew install --cask temurin@8
```

如果 Homebrew 没有提供 x86_64 的 cask，可以手动下载：
- 访问 https://adoptium.net/temurin/releases/?os=mac&arch=x64&package=jdk&version=8
- 下载 macOS x64 的 `.pkg` 安装包并安装

安装后验证：

```bash
/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/bin/java -version
# 应显示: OpenJDK ... x86_64
file /Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/bin/java
# 应显示: Mach-O 64-bit executable x86_64
```

> **注意**: ARM64 JDK（如 Oracle JDK 8 ARM64）不能使用，因为 LWJGL 原生库是 x86_64 的。

## 2. 创建 Conda 环境

```bash
conda create -n rocket2-arm64 python=3.10 pip -y
conda activate rocket2-arm64
```

> **必须是 Python 3.10+**，`minestudio==1.1.2` 要求 `Requires-Python >=3.10`。

验证环境：

```bash
which pip
# 应指向: .../envs/rocket2-arm64/bin/pip

python --version
# 应显示: Python 3.10.x
```

## 3. 安装 Python 依赖

由于 `minestudio` 的依赖中包含 `cuda-python` 等在 macOS 上不可用的包，
需要按特定顺序安装。

```bash
cd /path/to/rocket2

# 3.1 安装 PyTorch（ARM64 原生版，支持 MPS 加速）
pip install torch torchvision

# 3.2 安装 minestudio（跳过依赖解析，避免 cuda-python 冲突）
pip install minestudio==1.1.2 --no-deps

# 3.3 安装项目依赖（包含 minestudio 的运行时依赖）
pip install -r requirements.txt

# 3.4 安装 SAM-2（从本地 MineStudio 源码）
cd MineStudio/minestudio/utils/realtime_sam
pip install --no-build-isolation -e .
cd /path/to/rocket2
```

> `setup.py` 已修改为跨平台版本，在没有 CUDA 时自动跳过 CUDA 扩展编译，
> `sam2/utils/misc.py` 中的 `get_connected_components` 会自动 fallback 到 cv2 实现。

## 4. 配置 MineStudio 模拟器

### 4.1 设置持久化存储目录

MineStudio 默认将引擎下载到系统临时目录（`/var/folders/.../T/MineStudio`），
重启后会被清理。设置环境变量让它使用永久路径：

```bash
# 添加到 ~/.zshrc
echo 'export MINESTUDIO_DIR="$HOME/.minestudio"' >> ~/.zshrc
source ~/.zshrc
```

### 4.2 下载 LWJGL macOS 原生库

MineStudio 的引擎 JAR（`mcprec-6.13.jar`）只打包了 Linux 的 LWJGL 原生库，
需要手动下载 macOS x86_64 版本：

```bash
mkdir -p ~/.minestudio/lwjgl-macos-natives/lib

cd ~/.minestudio/lwjgl-macos-natives

LWJGL_VERSION="3.2.2"
BASE_URL="https://repo1.maven.org/maven2/org/lwjgl"

for module in lwjgl lwjgl-glfw lwjgl-openal lwjgl-opengl lwjgl-stb lwjgl-tinyfd lwjgl-jemalloc; do
    echo "Downloading ${module}..."
    curl -sL -O "${BASE_URL}/${module}/${LWJGL_VERSION}/${module}-${LWJGL_VERSION}-natives-macos.jar"
done

# 解压 .dylib 文件
for jar in *.jar; do
    unzip -o -j "$jar" "*.dylib" -d lib/ 2>/dev/null
done

echo "--- 验证 ---"
file lib/*.dylib
# 所有文件应显示: Mach-O 64-bit dynamically linked shared library x86_64
```

最终目录结构：

```
~/.minestudio/lwjgl-macos-natives/lib/
├── libglfw.dylib
├── libjemalloc.dylib
├── liblwjgl.dylib
├── liblwjgl_opengl.dylib
├── liblwjgl_stb.dylib
├── liblwjgl_tinyfd.dylib
└── libopenal.dylib
```

### 4.3 Patch minestudio 源码（2 个文件）

minestudio 的两个文件假设运行在 Linux + CUDA 环境，需要 patch 以支持 macOS。

#### Patch 1: gpu_utils.py

将 CUDA import 包在 try/except 中，macOS 上无 CUDA 时返回 `cpu`。

```bash
# 定位文件
SITE_PACKAGES=$(python -c "import minestudio; print(minestudio.__path__[0])")
GPU_UTILS="$SITE_PACKAGES/simulator/minerl/env/gpu_utils.py"

echo "Patching: $GPU_UTILS"
```

将文件内容替换为：

```python
'''
Date: 2024-11-29 11:05:35
LastEditors: caishaofei caishaofei@stu.pku.edu.cn
LastEditTime: 2024-11-29 11:07:37
FilePath: /MineStudio/minestudio/simulator/minerl/env/gpu_utils.py
'''
# https://nvidia.github.io/cuda-python/
import argparse
import os

try:
    from cuda import cuda, cudart
    _HAS_CUDA = True
except (ImportError, OSError):
    _HAS_CUDA = False

def call_and_check_error(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        assert result[0] == 0, f"cuda-python error, {result[0]}"
        if len(result) == 2:
            return result[1]
        else:
            assert len(result) == 1, "Unsupported function call"
            return None
    return wrapper

def getCudaDeviceCount():
    if not _HAS_CUDA:
        return 0
    return call_and_check_error(cudart.cudaGetDeviceCount)()

def getPCIBusIdByCudaDeviceOrdinal(cuda_device_id):
    '''
    cuda_device_id 在 0 ~ getCudaDeviceCount() - 1 之间取值，受到 CUDA_VISIBLE_DEVICES 影响
    '''
    device = call_and_check_error(cuda.cuDeviceGet)(cuda_device_id)
    result = call_and_check_error(cuda.cuDeviceGetPCIBusId)(100, device)
    return result.decode("ascii").split('\0')[0]

if __name__ == "__main__":
    if not _HAS_CUDA or os.environ.get("MINESTUDIO_GPU_RENDER", 0) != '1':
        print("cpu")
        exit(0)
    try:
        call_and_check_error(cuda.cuInit)(0)
    except:
        print("cpu")
        exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=str)
    args = parser.parse_args()
    index = int (args.index)
    num_cuda_devices = getCudaDeviceCount()
    if num_cuda_devices == 0:
        device = "cpu"
    else:
        cuda_device_id = index % num_cuda_devices
        pci_bus_id = getPCIBusIdByCudaDeviceOrdinal(cuda_device_id)
        device = os.path.realpath(f"/dev/dri/by-path/pci-{pci_bus_id.lower()}-card")
    print(device)
```

#### Patch 2: launchClient.sh

macOS 上跳过 `xvfb-run`（Linux 虚拟显示），使用 x86_64 JDK + LWJGL 原生库路径。

```bash
LAUNCH_SH="$SITE_PACKAGES/simulator/minerl/env/launchClient.sh"
echo "Patching: $LAUNCH_SH"
```

将文件内容替换为：

```bash
#!/bin/bash

replaceable=0
port=0
seed="NONE"
maxMem="2G"
device="egl"
fatjar=build/libs/mcprec-6.13.jar

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        -device) device="$2"; shift;;
        -fatjar) fatjar="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port <port>] [-seed <seed>] [-maxMem <maxMem>] [-device <device>] [-fatjar <fatjar>]"
            exit 1;;
    esac
    shift
done

if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi

if [ "$device" == "cpu" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS: use x86_64 JDK (Rosetta) + LWJGL macOS natives
        LWJGL_NATIVES="$HOME/.minestudio/lwjgl-macos-natives/lib"
        JAVA_X86="/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/bin/java"
        if [ -f "$JAVA_X86" ] && [ -d "$LWJGL_NATIVES" ]; then
            JAVA_CMD="$JAVA_X86"
            LWJGL_OPTS="-Dorg.lwjgl.librarypath=$LWJGL_NATIVES -XstartOnFirstThread"
        else
            JAVA_CMD="java"
            LWJGL_OPTS=""
        fi
        $JAVA_CMD $LWJGL_OPTS -Xmx$maxMem -jar $fatjar --envPort=$port
    else
        xvfb-run -a java -Xmx$maxMem -jar $fatjar --envPort=$port
    fi
else
    vglrun -d $device java -Xmx$maxMem -jar $fatjar --envPort=$port
fi

[ $replaceable -gt 0 ]
```

#### Patch 3: base_policy.py (MPS float64 兼容)

MPS 不支持 float64，minestudio 的 `_batchify` 方法需要在转设备前将 float64 转为 float32。

```bash
BASE_POLICY="$SITE_PACKAGES/models/base_policy.py"
echo "Patching: $BASE_POLICY"
```

找到 `_batchify` 方法，将其替换为：

```python
def _batchify(self, elem):
    if isinstance(elem, (int, float)):
        elem = torch.tensor(elem, dtype=torch.float32, device=self.device)
    if isinstance(elem, np.ndarray):
        t = torch.from_numpy(elem)
        if t.is_floating_point() and t.dtype == torch.float64:
            t = t.float()
        return t.unsqueeze(0).unsqueeze(0).to(self.device)
    elif isinstance(elem, torch.Tensor):
        if elem.is_floating_point() and elem.dtype == torch.float64:
            elem = elem.float()
        return elem.unsqueeze(0).unsqueeze(0).to(self.device)
    elif isinstance(elem, str):
        return [[elem]]
    else:
        return elem
```

### 4.4 一键 Patch 脚本

为了方便，可以使用以下脚本自动完成两个 patch：

```bash
#!/bin/bash
# patch_minestudio_macos.sh - 自动 patch minestudio 以支持 macOS

set -e

SITE_PACKAGES=$(python -c "import minestudio; print(minestudio.__path__[0])")

echo "=== Patching gpu_utils.py ==="
GPU_UTILS="$SITE_PACKAGES/simulator/minerl/env/gpu_utils.py"
python -c "
content = open('$GPU_UTILS').read()
if '_HAS_CUDA' in content:
    print('  Already patched, skipping.')
else:
    content = content.replace(
        'from cuda import cuda, cudart',
        '''try:
    from cuda import cuda, cudart
    _HAS_CUDA = True
except (ImportError, OSError):
    _HAS_CUDA = False'''
    )
    content = content.replace(
        'def getCudaDeviceCount():\n    return',
        'def getCudaDeviceCount():\n    if not _HAS_CUDA:\n        return 0\n    return'
    )
    content = content.replace(
        \"if os.environ.get(\\\"MINESTUDIO_GPU_RENDER\\\", 0) != '1':\",
        \"if not _HAS_CUDA or os.environ.get(\\\"MINESTUDIO_GPU_RENDER\\\", 0) != '1':\"
    )
    open('$GPU_UTILS', 'w').write(content)
    print('  Patched successfully.')
"

echo "=== Patching launchClient.sh ==="
LAUNCH_SH="$SITE_PACKAGES/simulator/minerl/env/launchClient.sh"
if grep -q 'Darwin' "$LAUNCH_SH"; then
    echo "  Already patched, skipping."
else
    sed -i '' 's|if \[ "\$device" == "cpu" \]; then|if [ "\$device" == "cpu" ]; then\
    if [[ "\$(uname)" == "Darwin" ]]; then\
        LWJGL_NATIVES="\$HOME/.minestudio/lwjgl-macos-natives/lib"\
        JAVA_X86="/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home/bin/java"\
        if [ -f "\$JAVA_X86" ] \&\& [ -d "\$LWJGL_NATIVES" ]; then\
            JAVA_CMD="\$JAVA_X86"\
            LWJGL_OPTS="-Dorg.lwjgl.librarypath=\$LWJGL_NATIVES -XstartOnFirstThread"\
        else\
            JAVA_CMD="java"\
            LWJGL_OPTS=""\
        fi\
        \$JAVA_CMD \$LWJGL_OPTS -Xmx\$maxMem -jar \$fatjar --envPort=\$port\
    else|' "$LAUNCH_SH"
    sed -i '' 's|    xvfb-run -a java -Xmx\$maxMem -jar \$fatjar --envPort=\$port|    xvfb-run -a java -Xmx\$maxMem -jar \$fatjar --envPort=\$port\
    fi|' "$LAUNCH_SH"
    echo "  Patched successfully."
fi

echo "=== Done ==="
```

## 5. 下载模拟器引擎并验证

```bash
conda activate rocket2-arm64
python -m minestudio.simulator.entry -y
```

首次运行会下载 ~458MB 引擎到 `~/.minestudio/`，随后启动 Minecraft 模拟器。

预期输出：

```
INFO: Starting Minecraft process with device: cpu
...
Speed Test Status:
Average FPS: ~25-28
Total Steps: 100
```

> GLFW error、OptiFine reflector 警告、Realms 授权警告均为正常现象，不影响运行。

## 6. 项目代码改动说明

以下改动已包含在项目仓库中，不需要手动操作：

### 6.1 SAM-2 setup.py (跨平台)

`MineStudio/minestudio/utils/realtime_sam/setup.py` 已修改为自动检测 CUDA：
- 有 CUDA + CUDA_HOME → 编译 CUDA C++ 扩展
- 无 CUDA → 跳过，使用 cv2 的 Python fallback

### 6.2 移除 sam2_wrapper.py

MineStudio 版 SAM-2 源码已经包含了所有设备兼容修复：
- `get_connected_components` 有 `try _C / except → cv2` fallback
- `_init_state` 使用 `next(self.parameters()).device`
- `_get_image_feature` / `_get_feature` 使用 `condition_state["device"]`

因此删除了冗余的 `sam2_wrapper.py`，`launch.py` 和 `interactive_eval.py`
改为直接 `from sam2.build_sam import build_sam2_camera_predictor`。

### 6.3 requirements.txt

- 添加了 `Pyro4>=4.82`（minestudio 的运行时依赖，`--no-deps` 安装时不会自动拉取）
- 包含了完整的 macOS 和 Linux 安装步骤注释

## 7. 故障排除

### Q: `Failed to locate library: liblwjgl.dylib`
确认 LWJGL 原生库已下载并解压到 `~/.minestudio/lwjgl-macos-natives/lib/`，
且 `launchClient.sh` 已 patch。

### Q: `xvfb-run: command not found`
确认 `launchClient.sh` 已 patch，macOS 上不使用 `xvfb-run`。

### Q: `No module named 'cuda'` / `gpu_utils.py` 报错
确认 `gpu_utils.py` 已 patch，macOS 上跳过 CUDA 导入。

### Q: `No module named 'Pyro4'`
```bash
pip install Pyro4
```

### Q: `OSError: CUDA_HOME environment variable is not set`（安装 SAM-2 时）
确认使用的是修改后的跨平台 `setup.py`（仓库中已包含）。

### Q: 引擎重启后需要重新下载
确认设置了 `MINESTUDIO_DIR` 环境变量：
```bash
export MINESTUDIO_DIR="$HOME/.minestudio"
```

### Q: 重新 pip install minestudio 后 patch 失效
minestudio 的 patch 是修改 site-packages 中的文件，重装后需要重新执行 4.3 节的 patch。

## 8. 架构总览

```
macOS ARM64 运行栈:

┌─────────────────────────────────────────────┐
│  Python (ARM64)                              │
│  ├── ROCKET-2 模型 (MPS/CPU 推理)            │
│  ├── SAM-2 (CPU, cv2 fallback)              │
│  ├── Gradio Web UI                          │
│  └── minestudio (patched for macOS)         │
│       └── Minecraft Java Process             │
│            ├── JDK 8 x86_64 (Rosetta 2)     │
│            └── LWJGL 3.2.2 macOS x86_64     │
└─────────────────────────────────────────────┘
```

## 9. 完整命令速查

```bash
# === 环境创建 ===
conda create -n rocket2-arm64 python=3.10 pip -y
conda activate rocket2-arm64

# === Python 依赖 ===
pip install torch torchvision
pip install minestudio==1.1.2 --no-deps
pip install -r requirements.txt
cd MineStudio/minestudio/utils/realtime_sam && pip install --no-build-isolation -e . && cd -

# === 环境变量 ===
echo 'export MINESTUDIO_DIR="$HOME/.minestudio"' >> ~/.zshrc
source ~/.zshrc

# === LWJGL 原生库 ===
mkdir -p ~/.minestudio/lwjgl-macos-natives/lib
cd ~/.minestudio/lwjgl-macos-natives
for m in lwjgl lwjgl-glfw lwjgl-openal lwjgl-opengl lwjgl-stb lwjgl-tinyfd lwjgl-jemalloc; do
    curl -sL -O "https://repo1.maven.org/maven2/org/lwjgl/$m/3.2.2/$m-3.2.2-natives-macos.jar"
done
for jar in *.jar; do unzip -o -j "$jar" "*.dylib" -d lib/ 2>/dev/null; done
cd -

# === Patch minestudio ===
# 参照 4.3 节手动 patch 或运行 patch 脚本

# === 验证 ===
python -m minestudio.simulator.entry -y
```

# CUDA Setup Instructions

由于uv sync会自动安装CPU版本的PyTorch，每次环境被重置后需要重新安装CUDA版本。

## 快速修复

当遇到 "Torch not compiled with CUDA enabled" 错误时，运行：

```powershell
.\setup_cuda.ps1
```

或手动运行：

```powershell
uv pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy==1.26.4
```

## 为什么会发生？

- `uv sync` 会从PyPI安装依赖，默认是CPU版本的torch
- VS Code或其他工具可能自动触发 `uv sync`
- 每次运行 `uv sync` 后都需要重新安装CUDA版本

## 验证CUDA是否可用

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 运行项目

```powershell
# 标准YOLO (80个COCO类别)
python .\main.py

# YOLO-World (112+自定义类别)
python .\main_world.py
```

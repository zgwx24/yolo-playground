# 安装CUDA版本的PyTorch
# 运行此脚本: .\setup_cuda.ps1

Write-Host "Installing CUDA-enabled PyTorch..." -ForegroundColor Green

# 激活虚拟环境（如果还没激活）
if (-not $env:VIRTUAL_ENV) {
    & .\.venv\Scripts\Activate.ps1
}

# 安装CUDA版本的PyTorch
uv pip install --force-reinstall torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 修复numpy版本
uv pip install numpy==1.26.4

# 验证安装
Write-Host "`nVerifying CUDA installation..." -ForegroundColor Green
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torch version: {torch.__version__}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`nSetup complete! You can now run: python .\main_world.py" -ForegroundColor Green

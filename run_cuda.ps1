# 自动确保CUDA环境的运行脚本
param(
    [string]$Script = "main_world.py"
)

# 添加uv到PATH
$env:Path = "C:\Users\zhougu\.local\bin;$env:Path"

# 检查CUDA是否可用
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null

if ($cudaCheck -ne "True") {
    Write-Host "CUDA not available, fixing environment..." -ForegroundColor Yellow
    & .\setup_cuda.ps1
}

# 运行脚本
Write-Host "Running $Script with CUDA..." -ForegroundColor Green
python $Script

@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo   Project-Alpha-02 环境安装
echo   Python 3.13 + PyTorch CUDA 13.0 (cu130) + LightGBM
echo ============================================================
echo.

cd /d "d:\.shigodo\shigodo\Quantification\Project-Alpha-02"

REM ========== Step 1: 升级 pip ==========
echo [1/4] 升级 pip (清华镜像)...
.venv\Scripts\python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
echo.

REM ========== Step 2: PyTorch cu130 走官方源 ==========
echo [2/4] 安装 PyTorch cu130 (1.9GB, 官方源)...
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if %errorlevel% neq 0 (
    echo [X] PyTorch 安装失败！
    pause
    exit /b 1
)
echo [OK] PyTorch cu130 安装完成
echo.

REM ========== Step 3: 其余依赖走清华镜像 ==========
echo [3/4] 安装其余依赖 (清华镜像)...
.venv\Scripts\pip install lightgbm pandas numpy scipy scikit-learn matplotlib akshare baostock tqdm jupyter -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
if %errorlevel% neq 0 (
    echo [X] 依赖安装失败！
    pause
    exit /b 1
)
echo [OK] 依赖安装完成
echo.

REM ========== Step 4: 验证环境 ==========
echo [4/4] 验证环境...
.venv\Scripts\python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); import lightgbm; print('LightGBM:', lightgbm.__version__); import pandas; print('Pandas:', pandas.__version__); print('ALL OK')"
if %errorlevel% neq 0 (
    echo [X] 环境验证失败！
    pause
    exit /b 1
)
echo.

echo ============================================================
echo   环境安装完成！
echo   PyTorch cu130 + 所有依赖就绪
echo ============================================================

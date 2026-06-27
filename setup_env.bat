@echo off
chcp 65001 >nul 2>&1
echo ============================================================
echo   Project-Beta 环境自动配置脚本
echo   复现招商证券《多模型集成量价Alpha策略》
echo   Python 3.13 + PyTorch CUDA 13.0 (cu130) + LightGBM
echo ============================================================
echo.

:: ---- Step 0: 检查虚拟环境 ----
echo [Step 0/6] 检查虚拟环境...
if not exist ".venv\Scripts\python.exe" (
    echo   [X] 未找到 .venv，请先创建或复制虚拟环境:
    echo       python -m venv .venv
    echo       或从 Project-Alpha 复制
    pause
    exit /b 1
)
echo   [OK] .venv 已存在
echo.

:: ---- Step 1: 配置 pip 国内镜像源（永久生效） ----
echo [Step 1/6] 配置 pip 国内镜像源（清华大学 TUNA）...
.venv\Scripts\pip.exe config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
.venv\Scripts\pip.exe config set global.trusted-host pypi.tuna.tsinghua.edu.cn
echo   [OK] 镜像源已配置
echo.

:: ---- Step 2: 升级 pip ----
echo [Step 2/6] 升级 pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
echo.

:: ---- Step 3: 安装 PyTorch CUDA 13.0（如果未安装） ----
echo [Step 3/6] 检查 PyTorch...
.venv\Scripts\python.exe -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>nul
if %errorlevel% neq 0 (
    echo   未检测到PyTorch，开始安装...
    .venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu130
    if %errorlevel% neq 0 (
        echo   [X] PyTorch 安装失败！请检查网络连接后重试
        pause
        exit /b 1
    )
)
echo   [OK] PyTorch 可用
echo.

:: ---- Step 4: 安装其余依赖 ----
echo [Step 4/6] 安装其余依赖（从清华镜像源下载）...
.venv\Scripts\pip.exe install pandas numpy scipy scikit-learn matplotlib akshare baostock lightgbm tqdm jupyter
if %errorlevel% neq 0 (
    echo   [X] 依赖安装失败！请检查网络连接后重试
    pause
    exit /b 1
)
echo   [OK] 所有依赖安装成功
echo.

:: ---- Step 5: 验证安装 ----
echo [Step 5/6] 验证安装结果...
.venv\Scripts\python.exe -c "import sys; print(f'  Python: {sys.version}'); import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  GPU: N/A'); import lightgbm; print(f'  LightGBM: {lightgbm.__version__}'); import pandas; print(f'  Pandas: {pandas.__version__}'); import numpy; print(f'  NumPy: {numpy.__version__}'); import scipy; print(f'  SciPy: {scipy.__version__}'); import sklearn; print(f'  Scikit-learn: {sklearn.__version__}')"
echo.

:: ---- Step 6: 创建目录 & 运行初始化 ----
echo [Step 6/6] 创建项目目录结构 & 初始化验证...
.venv\Scripts\python.exe scripts\init_project.py
echo.

echo ============================================================
echo   环境配置完成！
echo ============================================================
echo.
echo 下一步操作:
echo   1. .venv\Scripts\activate                # 激活虚拟环境
echo   2. python -m utils.data_loader           # 获取/复用数据
echo   3. python -m utils.feature_engine        # 量价特征工程
echo   4. python -m utils.preprocess            # 预处理
echo   5. python train.py --model mlp           # 训练MLP
echo   6. python train.py --model gbdt          # 训练LightGBM
echo   7. python train.py --model gru           # 训练GRU
echo   8. python ensemble.py                    # ICIR集成
echo   9. python evaluate.py                    # 评估
echo.
pause

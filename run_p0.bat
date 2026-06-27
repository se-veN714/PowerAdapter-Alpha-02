@echo off
chcp 65001 >nul
cd /d "d:\.shigodo\shigodo\Quantification\Project-Alpha-02"
echo ============================================================
echo   Project-Alpha-02 路线B P0: 数据管线
echo   50只股票 + 6字段(OHLC+VWAP+VOL) x 30天 = 180维截面特征
echo ============================================================
echo.

set PYTHONPATH=d:\.shigodo\shigodo\Quantification\Project-Alpha-02

echo [Step 0/2] 加载数据（计算VWAP + 衍生特征）...
.venv\Scripts\python -m utils.data_loader
if %errorlevel% neq 0 (
    echo [X] 数据加载失败！
    pause
    exit /b 1
)
echo.

echo [Step 1/2] 特征工程（30字段 x 30天滞后展开 + 序列构建）...
.venv\Scripts\python -m utils.feature_engine
if %errorlevel% neq 0 (
    echo [X] 特征工程失败！
    pause
    exit /b 1
)
echo.

echo [Step 2/2] 预处理（MAD+zscore+Label标准化）...
.venv\Scripts\python -m utils.preprocess
if %errorlevel% neq 0 (
    echo [X] 预处理失败！
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   P0 完成！数据管线结束
echo   输出: data/processed/processed_data.csv
echo ============================================================
pause

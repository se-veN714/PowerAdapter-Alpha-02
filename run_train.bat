@echo off
chcp 65001 >nul
cd /d "d:\.shigodo\shigodo\Quantification\Project-Alpha-02"
echo ============================================================
echo   Project-Alpha-02 路线B: 滚动训练
echo   4模型(MLP/GBDT/GRU/AGRU) x 6窗口 x 3种子 = 72任务
echo   配置: 180维量价特征 + VWAP 10天标签 + MSE Loss
echo   日志: log_writer 双路输出 (控制台 + 文件)
echo ============================================================
echo.

set PYTHONPATH=d:\.shigodo\shigodo\Quantification\Project-Alpha-02

echo [开始] 滚动训练（带log_writer包裹）...
.venv\Scripts\python train.py --mode rolling
if %errorlevel% neq 0 (
    echo [X] 训练失败！查看日志: wb-remote-monitor\logs\training_*.log
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   训练完成！
echo   日志: wb-remote-monitor\logs\training_*.log
echo   模型: checkpoints\*.pt / *.joblib
echo ============================================================
pause

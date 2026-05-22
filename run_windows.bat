@echo off
echo Building project...
gcc -std=c11 -Wall -Wextra -O2 src\main.c src\decision_tree.c src\dataset.c src\tree_visualization.c -lm -o decision_tree_robot.exe
if %errorlevel% neq 0 (
    echo Build failed. Check that GCC/MinGW is installed and added to PATH.
    pause
    exit /b %errorlevel%
)
echo Starting program...
.\decision_tree_robot.exe
pause

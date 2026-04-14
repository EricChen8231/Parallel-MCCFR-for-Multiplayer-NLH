@echo off
:: =============================================================================
:: build.bat — Build poker_cuda on Windows with MSVC + CUDA
:: Run from the poker_cuda\ directory using the
:: "x64 Native Tools Command Prompt for VS 2022"
:: =============================================================================

echo === Poker CUDA Build Script ===
echo.

:: Verify nvcc is available
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: nvcc not found. Please install CUDA Toolkit first.
    echo        Download: https://developer.nvidia.com/cuda-downloads
    echo        After installing, open a new terminal and re-run this script.
    pause
    exit /b 1
)

:: Verify cmake is available
where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: cmake not found. Install with: winget install Kitware.CMake
    echo        After installing, open a new terminal and re-run this script.
    pause
    exit /b 1
)

echo [1/3] Configuring with CMake...
cmake -B build ^
      -DCMAKE_BUILD_TYPE=Release ^
      -DCMAKE_CUDA_ARCHITECTURES=86
if errorlevel 1 (
    echo ERROR: CMake configuration failed.
    pause
    exit /b 1
)

echo.
echo [2/3] Building (this takes 1-3 minutes on first build)...
cmake --build build --config Release -j8
if errorlevel 1 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Checking for handranks.dat...
if not exist "data\handranks.dat" (
    echo WARNING: data\handranks.dat not found.
    echo          Download from: https://github.com/b-g-goodell/two-plus-two-hand-evaluator
    echo          Place it at: poker_cuda\data\handranks.dat
    echo          The binary will still be built — you can download it before running.
) else (
    echo          data\handranks.dat found.
)

echo.
echo === Build complete! ===
echo.
echo To verify your GPU:
echo   build\Release\poker_cuda.exe --mode info
echo.
echo To train (2 players, 1000 iterations):
echo   build\Release\poker_cuda.exe --mode train --players 2 --iters 1000 --batch 32768 --handranks data\handranks.dat
echo.
pause

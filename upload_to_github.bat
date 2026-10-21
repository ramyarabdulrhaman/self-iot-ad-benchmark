@echo off
echo =======================================================
echo Uploading self-iot-ad-benchmark to GitHub...
echo =======================================================
cd /d "%~dp0"

echo 1. Verifying Git commit...
git status

echo 2. Pushing to GitHub (forcing overwrite of empty remote README if present)...
git push -u origin main --force

echo =======================================================
if %ERRORLEVEL% equ 0 (
    echo SUCCESS! Your benchmark repository has been uploaded to GitHub!
    echo Visit: https://github.com/ramyarabdulrhaman/self-iot-ad-benchmark
) else (
    echo.
    echo If push failed due to authentication, please log in with your GitHub Personal Access Token or Browser.
)
echo =======================================================
pause

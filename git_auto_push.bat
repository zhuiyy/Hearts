@echo off
echo Adding files...
git add .

for /f "tokens=*" %%a in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"') do set DATETIME=%%a
echo Committing with message: "Auto update %DATETIME%"
git commit -m "Auto update %DATETIME%"

echo Pushing to remote...
git push

echo Done.
pause
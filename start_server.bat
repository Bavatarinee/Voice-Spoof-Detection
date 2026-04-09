@echo off
echo ============================================================
echo   Voice Spoof Detection - Backend Starter
echo   (Run as Administrator for firewall rule setup)
echo ============================================================
echo.

REM Add Windows Firewall rule to allow port 5000 from local network
echo [1/3] Adding Windows Firewall rule for port 5000...
netsh advfirewall firewall delete rule name="Flask Voice Spoof Port 5000" >nul 2>&1
netsh advfirewall firewall add rule name="Flask Voice Spoof Port 5000" dir=in action=allow protocol=TCP localport=5000
if %errorlevel%==0 (
    echo       OK - Firewall rule added.
) else (
    echo       WARNING - Could not add firewall rule. Try running this script as Administrator.
    echo       Right-click this file ^> Run as administrator
)

echo.
echo [2/3] Your local IP addresses:
ipconfig | findstr "IPv4"

echo.
echo [3/3] Starting Flask backend on port 5000...
echo       (Your phone must be on the SAME Wi-Fi / hotspot)
echo       Update the IP in the mobile app Settings tab.
echo.

cd /d "%~dp0"
python app.py

pause

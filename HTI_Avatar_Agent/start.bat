@echo off
title Meera - HTI Lab 3D Conversational Avatar Server
echo ============================================================
echo   Starting Meera - HTI Lab 3D Conversational Avatar Server
echo ============================================================
echo.
echo Opening browser at http://localhost:5173/web/index.html ...
start http://localhost:5173/web/index.html

echo.
echo Running Python Avatar Web Server on port 5173...
echo (Press Ctrl+C in this window to stop the server)
echo.
python server.py
pause

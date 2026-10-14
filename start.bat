@echo off
rem Double-click to set up (first run) and start AtheroVision.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*

@echo off
cd /d "%~dp0.."
set "THESEUS_PYTHON=%CD%\.venv-puffer\Scripts\python.exe"
if not exist "%THESEUS_PYTHON%" set "THESEUS_PYTHON=py"
"%THESEUS_PYTHON%" scripts\theseus_cli.py %*

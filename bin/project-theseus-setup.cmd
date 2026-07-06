@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=%CD%\.venv-puffer\Scripts\python.exe"
if not exist "%PYTHON%" (
  where python >nul 2>nul
  if errorlevel 1 (
    set "PYTHON=py"
  ) else (
    set "PYTHON=python"
  )
)
"%PYTHON%" scripts\theseus_setup_wizard.py --open

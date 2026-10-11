@echo off
setlocal
cd /d "%~dp0"
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
if not defined PY goto nopython
%PY% -c "import numpy, torch, astropy, fastapi, uvicorn" >nul 2>nul && goto ready
echo Installing Anelench into a local .venv folder (first run only, a few minutes)...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto failed
set "PY=.venv\Scripts\python.exe"
%PY% -m pip install --upgrade pip || goto failed
%PY% -m pip install -r requirements.txt || goto failed
:ready
%PY% scripts\download_eht_data.py || goto failed
%PY% -m anelench.server --open
goto end
:nopython
echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and run this file again.
pause
exit /b 1
:failed
echo Setup did not finish. Check the internet connection and the messages above.
pause
exit /b 1
:end
endlocal

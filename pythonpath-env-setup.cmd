@ECHO OFF

REM This Windows script is made to add or append the Polybench/Python project
REM root into the PYTHONPATH environment variable.
REM The PYTHONPATH environment variable is only set for the current terminal
REM session. If that session ends, changes made by this script will be lost.
REM
REM This script was tested on Windows 10 (1909), but should work on any Windows
REM NT version where the Python3 interpreter can run.
REM
REM The script asumes to be placed in the Polybench/Python folder's root.
REM
REM Author: Miguel Angel Abella Gonzalez    <miguel.abella@udc.es>


REM Create the auxiliary environment variable POLYBENCH_PYTHON_ROOT.
REM We will use this variable to determine the actual root directory for
REM Polybench/Python in case this script is run either from the actual root
REM directory or from a different path.
REM
REM From the terminal's documentation (HELP CALL) we can extract the path of any
REM argument passed to the script. Since arg(0) is the script name, we can use
REM it to determine the root path (asuming this script is there).
SET POLYBENCH_PYTHON_ROOT=%~dp0
REM Remove the trailing slash if present.
IF %POLYBENCH_PYTHON_ROOT:~-1%==\ SET POLYBENCH_PYTHON_ROOT=%POLYBENCH_PYTHON_ROOT:~0,-1%


REM Test the existence of the PYTHONPATH environment variable and perform the
REM required operations to update it if necessary.
IF NOT DEFINED PYTHONPATH (
    REM When the PYTHONPATH environment variable is not defined we create it.
    SET PYTHONPATH=%POLYBENCH_PYTHON_ROOT%
    ECHO Environment variable PYTHONPATH updated. Set: '%POLYBENCH_PYTHON_ROOT%'
) ELSE (
    REM When the PYTHONPATH environment variable is defined we need to test if
    REM the Polybench/Python root directory is already present or not.

    REM Use "findstr" as suggested here: https://stackoverflow.com/a/7218493
    ECHO.%PYTHONPATH% | findstr /C:%POLYBENCH_PYTHON_ROOT% 1>nul
    IF ERRORLEVEL 1 (
        REM ERRORLEVEL >= 1 -> substring not found. Append the root directory.
        SET PYTHONPATH=%PYTHONPATH%;%POLYBENCH_PYTHON_ROOT%
        ECHO Environment variable PYTHONPATH updated. Append: '%POLYBENCH_PYTHON_ROOT%'
    ) ELSE (
        REM ERRORLEVEL < 1 (probably 0) -> substring found. Nothing to do.
        ECHO Environment variable PYTHONPATH not updated. Already contains '%POLYBENCH_PYTHON_ROOT%'
    )
)

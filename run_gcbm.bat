@echo off

REM *************************** USER CONFIGURATION ***************************
REM Set simulation start and end years.
set SIMULATION_START_YEAR=2015
set SIMULATION_END_YEAR=2100

REM Set Python path - change this to your Python installation directory.
set GCBM_PYTHON=C:\Python37

REM Is your version of MS Access 32 (x86) or 64 (x64) bit?
set PLATFORM=x64
REM **************************************************************************

REM Set GDAL library paths.
set GDAL_BIN=%GCBM_PYTHON%\lib\site-packages\osgeo
set GDAL_DATA=%GDAL_BIN%\data\gdal
set PROJECT_DIR=C:\Users\k5dsouza\OneDrive - University of Waterloo\Documents\Forest_Sims\GCBM_Afforestation\spatially_referenced
set PROJ_LIB=%PROJECT_DIR%\tools\GCBM
set EXPS_DIR=%PROJECT_DIR%\layers\raw\exps

set PYTHONPATH=%GCBM_PYTHON%;%GCBM_PYTHON%\lib\site-packages
set PATH=%GCBM_PYTHON%;%GDAL_BIN%;%GDAL_DATA%;%GCBM_PYTHON%\scripts;%GCBM_PYTHON%\lib\site-packages
set GCBM_FOLDER_NAME=GCBM

REM Clean up log and output directories.
if exist logs rd /s /q logs
if exist processed_output rd /s /q processed_output
if exist gcbm_project\output rd /s /q gcbm_project\output
if exist layers\tiled rd /s /q layers\tiled
md logs
md processed_output
md gcbm_project\output
md layers\tiled

set /A exps=0
for /D %%a in ("%EXPS_DIR%\*") do (
    SET /A exps+=1
)

echo Preparing to run %exps% experiments in total
for /l %%x in (1, 1, %exps%) do (
    echo Preparing for new scenario...
    "%GCBM_PYTHON%\python.exe" src\prep.py --exp_num %%x

    REM Run the tiler to process spatial input data.
    pushd %GCBM_FOLDER_NAME%\layers\tiled
        echo Tiling spatial layers...
        "%GCBM_PYTHON%\python.exe" ..\..\tools\tiler\tiler.py > temp_out.txt
    popd

	set /p result=<layers\tiled\temp_out.txt
	setlocal EnableDelayedExpansion
	if !result!==2 (
		echo Cleaning post and storing outputs...
		"%GCBM_PYTHON%\python.exe" src\post.py --exp_num %%x
		endlocal
	) ELSE (
		endlocal

		REM Generate the GCBM input database from a yield table and CBM3 ArchiveIndex database.
		echo Generating GCBM input database...
		%GCBM_FOLDER_NAME%\tools\recliner2gcbm-%PLATFORM%\recliner2gcbm.exe -c %GCBM_FOLDER_NAME%\input_database\recliner2gcbm_config.json

		REM Configure GCBM.
		echo Updating GCBM configuration...
		"%GCBM_PYTHON%\python.exe" %GCBM_FOLDER_NAME%\tools\tiler\update_gcbm_config.py --layer_root %GCBM_FOLDER_NAME%\layers\tiled --template_path %GCBM_FOLDER_NAME%\gcbm_project\templates --output_path %GCBM_FOLDER_NAME%\gcbm_project --input_db_path %GCBM_FOLDER_NAME%\input_database\gcbm_input.db --start_year %SIMULATION_START_YEAR% --end_year %SIMULATION_END_YEAR% --log_path logs

		REM Run the GCBM simulation.
		pushd %GCBM_FOLDER_NAME%\gcbm_project
			echo Running GCBM...
			..\tools\GCBM\moja.cli.exe --config_file gcbm_config.cfg --config_provider provider_config.json
		popd

		REM Create the results database from the raw simulation output.
		echo Compiling results database...
		"%GCBM_PYTHON%\python.exe" %GCBM_FOLDER_NAME%\tools\compilegcbmresults\compileresults.py --results_db sqlite:///%GCBM_FOLDER_NAME%\gcbm_project/output/gcbm_output.db --output_db sqlite:///%GCBM_FOLDER_NAME%\processed_output/compiled_gcbm_output.db

		echo Cleaning post and storing outputs...
		"%GCBM_PYTHON%\python.exe" src\post.py --exp_num %%x
	)
)

echo Done!
pause
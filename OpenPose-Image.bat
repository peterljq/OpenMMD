@echo off
cd /d %~dp0

rem ---  access the image

echo Please input the path of image

set INPUT_IMAGE=

set /P INPUT_IMAGE=■the path of IMAGE: 

rem echo INPUT_IMAGE：%INPUT_IMAGE%



IF /I "%INPUT_IMAGE%" EQU "" (

    ECHO Input error, the analysis terminates.

    EXIT /B

)



rem ---  Max number of people in the IMAGE

echo --------------

echo Max number of person in the IMAGE

echo Press "enter" to set the default: 1 person

set NUMBER_PEOPLE_MAX=1

set /P NUMBER_PEOPLE_MAX="Max number of people in the IMAGE: "



rem --echo NUMBER_PEOPLE_MAX: %NUMBER_PEOPLE_MAX%



rem -----------------------------------

rem --- IMAGE input

FOR %%1 IN (%INPUT_VIDEO%) DO (

    set INPUT_IMAGE_DIR=%%~dp1

    set INPUT_IMAGE_FILENAME=%%~n1

)





set DT=%date%
set TM=%time%
set TM2=%TM: =0%
set DTTM=%dt:~0,4%%dt:~5,2%%dt:~8,2%_%TM2:~0,2%%TM2:~3,2%%TM2:~6,2%
echo --------------
rem ------------------------------------------------

rem -- output json files
set OUTPUT_JSON_DIR=%INPUT_IMAGE_DIR%_json

rem echo %OUTPUT_JSON_DIR%



mkdir %OUTPUT_JSON_DIR%

echo JSON files will be outputted: %OUTPUT_JSON_DIR%



rem ------------------------------------------------

set OUTPUT_IMAGE_PATH=%INPUT_IMAGE_DIR%_openpose.png
echo Output PNG files：%OUTPUT_IMAGE_PATH%



echo --------------

echo Openpose started.

echo Press ESC to break the process.

echo --------------




bin\OpenPoseDemo.exe --video %INPUT_IMAGE% --write_json %OUTPUT_JSON_DIR% --write_video %OUTPUT_IMAGE_PATH% --number_people_max %NUMBER_PEOPLE_MAX% 



echo --------------

echo Done!

echo Openpose analysis finished.

echo Next step: using 3d-pose-baseline-vmd to process the JSON files.

echo %OUTPUT_JSON_DIR%

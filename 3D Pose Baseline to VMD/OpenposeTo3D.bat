es (40 sloc) 1.62 KB 

@echo off

rem --- 

rem --- 

rem --- 



cd /d %~dp0



rem ---  JSON

echo Please input the path of result from OpenPose Execution: JSON folder

echo Input is limited to English characters and numbers.

set OPENPOSE_JSON=

set /P OPENPOSE_JSON=■the path of result from OpenPose Execution (JSON folder): 

rem echo OPENPOSE_JSON：%OPENPOSE_JSON%



IF /I "%OPENPOSE_JSON%" EQU "" (

    ECHO the path you input is invalid. The runtime error occurs.

    EXIT /B

)



rem ---  The max number of people in your video



echo --------------

echo The max number of people in your video.

echo If no input and press Enter, the number of be set to default: 1 person.

set PERSON_IDX=1

set /P PERSON_IDX="The max number of people in your video: "



rem --echo PERSON_IDX: %PERSON_IDX%






echo --------------

echo If you want the detailed information of GIF, input yes.
echo If no input and press Enter, the generation setting of GIF will be set to default.
echo warn If you input warn, then no GIF will be generated.

set VERBOSE=2

set IS_DEBUG=no

set /P IS_DEBUG="the detailed information[yes/no/warn]: "



IF /I "%IS_DEBUG%" EQU "yes" (

    set VERBOSE=3

)



IF /I "%IS_DEBUG%" EQU "warn" (

    set VERBOSE=1

)



python src/openpose_3dpose_sandbox_vmd.py --camera_frame --residual --batch_norm --dropout 0.5 --max_norm --evaluateActionWise --use_sh --epochs 200 --load 4874200 --gif_fps 30 --verbose %VERBOSE% --openpose %OPENPOSE_JSON% --person_idx %PERSON_IDX%




@echo off

cd /d %~dp0

echo Please input the result folder path of the execution on 3d-pose-baseline-vmd.

set TARGET_DIR=
set /P TARGET_DIR=■3d-pose-baseline-vmd's execution result folrder path: 
rem echo TARGET_DIR：%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO Error input. Program terminates.
    EXIT /B
)


echo --------------
set MODEL_BONE_CSV=born\あにまさ式ミクボーン.csv
echo Enter the csv bone structure file path. Or press Enter to set to embedded Default (Miku_Bone_Structure.csv): あにまさ式ミクボーン.csv
set /P MODEL_BONE_CSV="the csv bone structure file: "



echo --------------
echo Input yes/no to let the output vmd file with IK/FK foot setting. 
echo Or press Enter to set to embedded Default: IK foot setting.
set IK_FLAG=1
set IS_IK=yes
set /P IS_IK="IK/FK[yes/no]: "

IF /I "%IS_IK%" EQU "no" (
    set IK_FLAG=0
    set HEEL_POSITION=0
    goto CONFIRM_CENTER
)


echo --------------
set HEEL_POSITION=0
echo Heel's position on Y axis. That is, the distance from the ground to the heel.
echo Press Enter to set to embedded Default: 0.
set /P HEEL_POSITION="Heel's position on Y axis: "


:CONFIRM_CENTER

echo --------------
set CENTER_XY_SCALE=30
echo X and Y axis mapping scale.
echo If you are not understanding this term, press Enter to set to embedded Default: 30 times.
set /P CENTER_XY_SCALE="X and Y axis mapping scale.: "

echo --------------
set CENTER_Z_SCALE=2
echo Z axis mapping scale.
echo If you are not understanding this term, press Enter to set to embedded Default: 2 times.
set /P CENTER_Z_SCALE="Z axis mapping scale: "


echo --------------
set GROBAL_X_ANGLE=15
echo After the 3-D convertion, the global angle/slope on X axis (-180 to 180)
echo If you are not understanding this term, press Enter to set to embedded Default: 15 degrees.
set /P GROBAL_X_ANGLE="Global X angle correction: "


echo --------------
set SMOOTH_TIMES=1
echo Smooth the output.
echo Larger your input is, smoother the actions are.
echo If you are not understanding this term, press Enter to set to embedded Default: 1 time.
set /P SMOOTH_TIMES="Smooth Times: "


echo --------------
set CENTER_DECIMATION_MOVE=0
echo Center Decimation.
echo If you are not understanding this term, press Enter to set to embedded Default: 0 decimation move.
set /P CENTER_DECIMATION_MOVE="decimation move: "

IF /I "%CENTER_DECIMATION_MOVE%" EQU "0" (
    set IK_DECIMATION_MOVE=0
    set DECIMATION_ANGLE=0
    set ALIGNMENT=1
    

    goto CONFRIM_LOG
)


echo --------------
set IK_DECIMATION_MOVE=1.5
echo IK Decimation.
echo If you are not understanding this term, press Enter to set to embedded Default: 1.5 times.
set /P IK_DECIMATION_MOVE="IK Decimation: "


echo --------------
set DECIMATION_ANGLE=10
echo Decimation angle (-180 to 180, integer)
echo If you are not understanding this term, press Enter to set to embedded Default: 10 degrees.
set /P DECIMATION_ANGLE="Decimation angle: "


echo --------------
echo Alignment (yes/no).
echo If you are not understanding this term, press Enter to set to embedded Default: the output will have default alignment.
set ALIGNMENT=1
set IS_ALIGNMENT=yes
set /P IS_ALIGNMENT="Alignment[yes/no]: "

IF /I "%IS_ALIGNMENT%" EQU "no" (
    set ALIGNMENT=0
)


:CONFRIM_LOG

echo --------------
echo Output verbose logs (yes/no).
echo If you are not understanding this term, press Enter to set to embedded Default: the detailed verbose logs will be outputted.
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="Execution logs[yes/no]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)


python applications\pos2vmd_multi.py -v %VERBOSE% -t "%TARGET_DIR%" -b %MODEL_BONE_CSV% -c %CENTER_XY_SCALE% -z %CENTER_Z_SCALE% -x %GROBAL_X_ANGLE% -m %CENTER_DECIMATION_MOVE% -i %IK_DECIMATION_MOVE% -d %DECIMATION_ANGLE% -a %ALIGNMENT% -k %IK_FLAG% -e %HEEL_POSITION%

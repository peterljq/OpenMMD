@echo off


cd /d %~dp0

echo Analysing the object depth from the video

set INPUT_VIDEO=
set /P INPUT_VIDEO=■Your video path: 
rem echo INPUT_VIDEO：%INPUT_VIDEO%

IF /I "%INPUT_VIDEO%" EQU "" (
    ECHO Error input. Program terminates.
    EXIT /B
)


echo Please input the result folder after the execution of 3d-pose-baseline-vmd model.
set TARGET_BASELINE_DIR=
set /P TARGET_BASELINE_DIR=■3D keypoints extraction result path: 
rem echo TARGET_DIR：%TARGET_DIR%

IF /I "%TARGET_BASELINE_DIR%" EQU "" (
    ECHO Error input. Program terminates.
    EXIT /B
)


echo --------------
set DEPTH_INTERVAL=10
echo Please set the depth interval. Smaller it is, clearer the results are.
echo Press Enter to set to Default: depth = 10.
set /P DEPTH_INTERVAL="Depth interval: "


echo --------------
echo Please input yes or no to decide whether you want to debug.
echo Press Enter to set to default Debug Mode.
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="[yes/no]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

python tensorflow/predict_video.py --model_path tensorflow/data/NYU_FCRN.ckpt --video_path %INPUT_VIDEO% --baseline_path %TARGET_BASELINE_DIR% --interval %DEPTH_INTERVAL% --verbose %VERBOSE%


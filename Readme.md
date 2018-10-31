OpenMMD
------

[OpenMMD](https://github.com/peterljq/OpenMMD) represents the OpenPose-based application that can directly **convert real-person videos to the motion of animation models (i.e. Miku, Anmicius)**. 

<p align="center">
<img src="/_pages/projects/OpenMMD-Anmicius-Static.jpg" width="240">
</p>

<p align="center">Anmicius 3D model</p>

## Features
The application is a combination of multiple Deep Learning Models with that the previous's output will be the next's input. 
- **Functionality**:
    - **3D single-person keypoints detection**:
        - Proposed by Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little. An effective baseline for 3d human pose estimation. In ICCV, 2017.
        - Recoded real-person video input and easy JSON files collections output.
    - **Video depth prediction**:
        - Proposed by Iro Laina, Christian Rupprecht. FCRN: Deeper Depth Prediction with Fully Convolutional Residual Networks.
        - Estimation of depth for objects, backgrounds and the moving person in the video (e.g. dancer).
    - **Human Motion keypoints to VMD motion files for MMD build** 
        - Proposed by Denis Tome, Chris Russell and Lourdes Agapito from UCL. Lifting from the Deep: Convolutional 3D Pose Estimation from a single image.
        - Edited by @miu200521358 to output VMD files for implementation of MMD videos.
- **Input**: videos of common formats (AVI, WAV) or images of common formats (PNG, JPG), 
- **Output**: Animations or Posetures of 3D models (i.e. Miku Dancing)
- **OS**: Windows (8, 10)

## Example Presentation
### I. Input Waving and Jumping
<p align="center">
    <img src="", width="360">
</p>

### II. Extraction of 3D keypoints
<p align="center">
    <img src="/_pages/projects/OpenMMD_smoothing.gif" width="240">
</p>

### III. Video Depth Prediction
<p align="center">
    <img src="/_pages/projects/OpenMMD_depth.gif" width="240">
</p>

### IV. Output VMD files and construct to MMD animations
<p align="center">
    <img src="/_pages/projects/OpenMMD-Anmicius.gif" width="240">
</p>


## Installation and Uninstallation
**Download the full pack**: 
- Simply download the whole pack contains the pre-trained models with optimized parameters and corresponding compilable codes. 

- [Google Drive](https://google.com)
- [Baidu Netdisk (百度网盘)](https://baidu.com)

**Follow the instruction to begin your first animation**: 
- Record a piece of video contains human motions. Satisfy all the prerequisite environment stated below.
- After downloading, firstly activate tensorflow environment in the terminal.
- Run OpenposeVideo.bat and follow the pop-out instructions.
- Then proceed to the 3d-pose-baseline-vmd folder and run OpenposeTo3D.bat. Follow the pop-out instructions.
- After that, proceed to the FCRN-DepthPrediction-vmd folder and run VideoToDepth.dat.
- Finally, proceed to the VMD-3d-pose-baseline-multi folder and run 3DToVMD.bat. You will get the vmd file.
- VMD files are 3D animation file used by MikuMikuDance, a program used to create dance animation movies. Open your MikuMikuDance and input the VMD file.
- You will see your Miku begin acting the same motions as that in your recorded video.

## Library dependencies
- OpenCV and relevance
```
pip install opencv-python
pip install numpy
pip install matplotlib
```
- Tensorflow and h5py. Please implement them in **anaconda**.
```
pip install tensorflow
conda create -n tensorflow pip python=3.6 // Create a workbench
activate tensorflow
pip install --ignore-installed --upgrade tensorflow
conda install h5py
//You will also need to install Keras in anaconda.

```
- Other libraries includes: 
```
pip install python-dateutil
pip install pytz
pip install pyparsing
pip install six
pip install imageio
```

## Feel free to give feedback!
My application is an open source project. Let me know if...

1. you find videos or images conversion does not work well.
2. you know how to speed up or improve the performance of 3D converting.
3. you have suggestions about future possible functionalities.
4. and anything you want to ask...

Just comment on GitHub or make a pull request and I will answer as soon as possible!

If you appreciate the project, please kindly star it. Feel free to fork it and develop your own 3D animations! Thank you!

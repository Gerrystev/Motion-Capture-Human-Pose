# Motion Capture using Human Pose Estimation 3D with YOLOv4-Tiny, EfficientNet Simple Baseline and VideoPose3D

This project has goal to substitute motion capture with human pose estimation with only deep learning methods and camera. This project using deep learning model such as YOLOv4-Tiny, EfficientNet Simple Baseline Modification, and VideoPose3D. These models can be view in the repository as link below: 
[YOLO-Darknet](https://github.com/AlexeyAB/darknet)
[PyTorch-YOLOv4](https://github.com/Tianxiaomo/pytorch-YOLOv4)
[Simple Baseline](https://github.com/microsoft/human-pose-estimation.pytorch)
[VideoPose3D](https://github.com/facebookresearch/VideoPose3D)

## EfficientNet Simple Baseline Modification
In this project using modification of [Simple Baseline](https://github.com/microsoft/human-pose-estimation.pytorch) to estimate 2D Human Pose Estimation. Here we using EfficientNet-B0 as backbone instead of ResNet from its original. From 140 training epoch, and evaluation from MPII dataset, we get following PCKh@0.5 result:
| | ResNet-152 | EfficientNet-B0 | EfficientNet-B5 |
|--|--|--|--|
| Head|97.03|95.71  |96.28  |
| Shoulder| 95.94 |94.24  |94.72  |
| Elbow|90.05 |85.82  |86.84  |
| Wrist|84.98  |79.13  |80.53  |
| Hip|89.16  |85.53  |86.64  |
| Knee|85.31  |80.27  |81.56  |
| Ankle|81.27|76.07 |77.23 |
| Mean|89.62  |85.93  |86.89  |
| Mean@0.1|89.62 |85.93 |86.89  |
| Train time|29.17 hours |25.6 hours |18.6 hours  |
| Process time|5.15 ms |4.54 ms |15.06 ms  |

As we can see, its original have better accuracy than its modification, but EfficientNet is faster for training and its process. For training we are using Tesla P100.

## Demonstration
In this application, we can insert video or livestream IP camera to be used for human pose estimation. This app very easy to use, we just need to choose video file contains the movement of a single person and start record.
![](https://github.com/Gerrystev/Motion-Capture-Human-Pose/blob/mpii/assets/MPII%20-%20Walk.gif)

## Note
For using model, extract from [this](https://drive.google.com/file/d/1PvRAveVK1TcJMM4zkeT3ffMCOp3CZeIZ/view?usp=sharing) to root folder

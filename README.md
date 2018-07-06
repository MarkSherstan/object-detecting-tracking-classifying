# object-detecting-tracking-classifying

Winter 2018 MecE 467 (Modeling and Simulation of Engineering Systems) final project. The objective of the following project was to create a MATLAB program that has the ability to detect, track, and recognize an object using a webcam. The program was to run in real time and in a variety of different settings. The final project objectives were as follows:
* Define a discrete event system carefully analyzing the inputs and outputs for each processing step.
* Use object detection algorithms to locate an area of interest in a video frame. 
* Efficiently track an object in a live video feed using computer vision and tracking algorithms.
* Apply deep learning for feature extraction and classification. 
* Minimize computational strain when using deep learning. 

## Getting Started

MATLAB 2017b and a Logitech C170 Webcam with the following MATLAB add ons were used:
* Neural Network Toolbox (Version 11.0)
* Statistics and Machine Learning Toolbox (Version 11.2)
* Parallel Computing Toolbox (Version 6.11)
* Neural Network Toolbox Model for AlexNet Network (Version 17.2.0)
* Control System Toolbox (Version 10.3)
* Image Processing Toolbox (Version 10.1)
* Image Acquisition Toolbox (Version 5.3)
* Computer Vision System Toolbox (Version 8.0)
* Image Acquisition Toolbox Support package for OS Generic Video Interface (Version 17.2.0)

Please also ensure to be running a CUDA-enabled NVIDIA GPU(s) with compute capability 3.0 or higher for training the network.  

### Running

Run the following to train the network. 

```
deepNetworkAlex.m
```

Run the following once network is trained to detect, track, and classify objects with a webcam.  

```
webcamKalmanFilterNet.m
```

## Other

Contact mshersta@ualberta.ca for trained network, images, full report, or any other questions and concerns. 

Video of final project is located here: https://www.youtube.com/watch?v=W8fdp0ZjQ5M 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

All credit and references can be found at the end of each .m file

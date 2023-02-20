## Course 4: Convolutional Neural Networks
The fourth course in the Deep Learning Specialization focuses on understanding how computer vision has evolved and becoming familiar with its exciting applications, such as autonomous driving, face recognition, reading radiology images, and more.

### Week 1: Foundations of Convolutional Neural Networks
Implement the foundational layers of CNNs (pooling, convolutions) and stack them properly in a deep network to solve multi-class image classification problems.

- Explain the convolution operation
- Apply two different types of pooling operations
- Identify the components used in a convolutional neural network (padding, stride, filter, ...) and their purpose
- Build a convolutional neural network
- Implement convolutional and pooling layers in numpy, including forward and backpropagation
- Implement helper functions to use when implementing a TensorFlow model
- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API
- Build and train a ConvNet in TensorFlow for a binary classification problem
- Build and train a ConvNet in TensorFlow for a multiclass classification problem
- Explain different use cases for the Sequential and Functional APIs

[Lecture Notes][L1]  
[Assignment: Convolutional Model: step by step][C4W1A1]  
[Assignment: Convolutional Neural Networks: Application][C4W1A2]  

### Week 2: Deep Convolutional Models: Case Studies
Discover practical techniques and methods used in research papers to apply transfer learning to a deep CNN.

- Implement the basic building blocks of ResNets in a deep neural network using Keras
- Train a state-of-the-art neural network for image classification
- Implement a skip connection on the network
- Create a dataset from a directory
- Preprocess and augment data using the Keras Sequential API
- Adapt a pretrained model to new data and train a classifier using the Functional API and MobileNet
- Fine-tune a classifier's final layers to improve accuracy

[Lecture Notes][L2]  
[Assignment: Residual Networks][C4W2A1]  
[Assignment: Transfer Learning with MobileNetV2][C4W2A2]  

### Week 3: Object Detection and Image Segmentation
Apply CNNs to computer vision: object detection and semantic segmentation using self-driving car datasets.

- Identify the components used for object detection (landmark, anchor, bounding box, grid, ...) and their purpose
- Implement object detection
- Implement non-max suppression to increase accuracy
- Implement intersection over union
- Handle bounding boxes, a type of image annotation popular in deep learning
- Apply sparse categorical crossentropy for pixelwise prediction
- Implement semantic image segmentation on the CARLA self-driving car dataset
- Explain the difference between a regular CNN and a U-net
- Build a U-Net

[Lecture Notes][L3]  
[Assignment: Object detection with YOLO][C4W3A1]  
[Assignment: Image Segmentation with U-Net][C4W3A2]  

### Week 4: Special Applications: Face Recognition and Neural Style Transfer
Discover how CNNs can be applied to multiple fields, including art generation and face recognition, and implement an algorithm to generate art and recognize faces.

- Differentiate between face recognition and face verification
- Implement one-shot learning to solve a face recognition problem
- Apply the triplet loss function to learn a network's parameters in the context of face recognition
- Explain how to pose face recognition as a binary classification problem
- Map face images into 128-dimensional encodings using a pretrained model
- Perform face verification and face recognition with these encodings
- Implement the Neural Style Transfer algorithm
- Generate novel artistic images using Neural Style Transfer
- Define the style cost function for Neural Style Transfer
- Define the content cost function for Neural Style Transfer

[Lecture Notes][L4]  
[Assignment: Face Recognition][C4W4A1]  
[Assignment: Deep Learning & Art: Neural Style Transfer][C4W4A2]  

### Reference
[Coursera - Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)

[L1]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W1-Foundations-of-Convolutional-Neural-Networks/C4_W1.pdf
[L2]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W2-Deep-Convolutional-Models-Case-Studies/C4_W2.pdf
[L3]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W3-Object-Detection/C4_W3.pdf
[L4]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W4-Special-Applications-Face-recognition-and-Neural-Style-Transfer/C4_W4.pdf

[C4W1A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W1-Foundations-of-Convolutional-Neural-Networks/A1/Convolution_model_Step_by_Step_v1.ipynb
[C4W1A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W1-Foundations-of-Convolutional-Neural-Networks/A2/Convolution_model_Application.ipynb
[C4W2A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W2-Deep-Convolutional-Models-Case-Studies/A1/Residual_Networks.ipynb
[C4W2A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W2-Deep-Convolutional-Models-Case-Studies/A2/Transfer_learning_with_MobileNet_v1.ipynb
[C4W3A1]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W3-Object-Detection/A1/Autonomous_driving_application_Car_detection.ipynb
[C4W3A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W3-Object-Detection/A2/Image_segmentation_Unet_v2.ipynb
[C4W4A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W4-Special-Applications-Face-recognition-and-Neural-Style-Transfer/A1/Face_Recognition.ipynb
[C4W4A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C4-Convolutional-Neural-Networks/W4-Special-Applications-Face-recognition-and-Neural-Style-Transfer/A2/Art_Generation_with_Neural_Style_Transfer.ipynb

## Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
The second course in the Deep Learning Specialization focuses on opening the black box of deep learning to understand the processes that drive performance and systematically generate good results.

### Week 1: Practical Aspects of Deep Learning
Discover and experiment with various initialization methods, apply L2 regularization and dropout to avoid model overfitting, and use gradient checking to identify errors in a fraud detection model.

- Give examples of how different types of initializations can lead to different results
- Examine the importance of initialization in complex neural networks
- Explain the difference between train/dev/test sets
- Diagnose the bias and variance issues in a model
- Assess the right time and place for using regularization methods such as dropout or L2 regularization
- Explain Vanishing and Exploding gradients and how to deal with them
- Use gradient checking to verify the accuracy of a backpropagation implementation
- Apply zeros initialization, random initialization, and He initialization
- Apply regularization to a deep learning model

[Lecture Notes][L1]  
[Assignment: Initialization][C2W1A1]  
[Assignment: Regularization][C2W1A2]  
[Assignment: Gradient Checking][C2W1A3]  

### Week 2: Optimization Algorithms
Extend the deep learning toolbox with advanced optimizations, random mini-batching, and learning rate decay scheduling to accelerate models.

- Apply optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
- Use random minibatches to accelerate convergence and improve optimization
- Describe the benefits of learning rate decay and apply it to the optimization

[Lecture Notes][L2]  
[Assignment: Optimization Methods][C2W2A1]  

### Week 3: Hyperparameter tuning, Batch Normalization, and Programming Frameworks
Explore TensorFlow, a deep learning framework that allows users to quickly and easily build neural networks; train a neural network on a TensorFlow dataset.

- Master the process of hyperparameter tuning
- Describe softmax classification for multiple classes
- Apply batch normalization to make a neural network more robust
- Build a neural network in TensorFlow and train it on a TensorFlow dataset
- Describe the purpose and operation of GradientTape
- Use tf.Variable to modify the state of a variable
- Apply TensorFlow decorators to speed up code
- Explain the difference between a variable and a constant

[Lecture Notes][L3]   
[Assignment: TensorFlow Tutorial][C2W3A1]    

### Reference
[Coursera - Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)


[C2W1A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W1-Practical-Aspects-of-Deep-Learning/A1/Initialization.ipynb
[C2W1A2]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W1-Practical-Aspects-of-Deep-Learning/A2/Regularization.ipynb
[C2W1A3]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W1-Practical-Aspects-of-Deep-Learning/A3/Gradient_Checking.ipynb
[C2W2A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W2-Optimization-Algorithms/A1/Optimization_methods.ipynb
[C2W3A1]: https://nbviewer.jupyter.org/github/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W3-Hyperparameter-Tuning-Batch-Normalization-and-Programming-Frameworks/A1/Tensorflow_introduction.ipynb

[L1]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W1-Practical-Aspects-of-Deep-Learning/C2_W1.pdf
[L2]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W2-Optimization-Algorithms/C2_W2.pdf
[L3]: https://github.com/pabaq/Coursera-Deep-Learning-Specialization/blob/main/C2-Improving-Deep-Neural-Networks-Hyperparameter-Tuning-Regularization-and-Optimization/W3-Hyperparameter-Tuning-Batch-Normalization-and-Programming-Frameworks/C2_W3.pdf
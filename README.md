# AI Programming with Python Project

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below.

![](assets/Flowers.png) 

## Installation
This project requires **Python 3.6.0** and the following Python libraries installed:
- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [PyTorch](https://pytorch.org/)


# Define a Convolution Neural Network.

To build a neural network with PyTorch, you'll use the torch.nn package. This package contains modules, extensible classes and all the required components to build neural networks.

Here, you'll build a basic convolution neural network (CNN) to classify the images from the CIFAR10 dataset.

A CNN is a class of neural networks, defined as multilayered neural networks designed to detect complex features in data. They're most commonly used in computer vision applications.

# Our network will be structured with the following 14 layers:
'''
Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> Linear.
The convolution layer
'''

The convolution layer is a main layer of CNN which helps us to detect features in images. Each of the layers has number of channels to detect specific features in images, and a number of kernels to define the size of the detected feature. Therefore, a convolution layer with 64 channels and kernel size of 3 x 3 would detect 64 distinct features, each of size 3 x 3. When you define a convolution layer, you provide the number of in-channels, the number of out-channels, and the kernel size. The number of out-channels in the layer serves as the number of in-channels to the next layer.

For example: A Convolution layer with in-channels=3, out-channels=10, and kernel-size=6 will get the RGB image (3 channels) as an input, and it will apply 10 feature detectors to the images with the kernel size of 6x6. Smaller kernel sizes will reduce computational time and weight sharing.
Other layers

# The following other layers are involved in our network:

*    The ReLU layer is an activation function to define all incoming features to be 0 or greater. When you apply this layer, any number less than 0 is changed to zero, while others are kept the same.
*   the BatchNorm2d layer applies normalization on the inputs to have zero mean and unit variance and increase the network accuracy.
*  The MaxPool layer will help us to ensure that the location of an object in an image will not affect the ability of the neural network to detect its specific features.
*    The Linear layer is final layers in our network, which computes the scores of each of the classes. In the CIFAR10 dataset, there are ten classes of labels. The label with the highest score will be the one model predicts. In the linear layer, you have to specify the number of input features and the number of output features which should correspond to the number of classes.

# How does a Neural Network work?

The CNN is a feed-forward network. During the training process, the network will process the input through all the layers, compute the loss to understand how far the predicted label of the image is falling from the correct one, and propagate the gradients back into the network to update the weights of the layers. By iterating over a huge dataset of inputs, the network will “learn” to set its weights to achieve the best results.

A forward function computes the value of the loss function, and the backward function computes the gradients of the learnable parameters. When you create our neural network with PyTorch, you only need to define the forward function. The backward function will be automatically defined.

# Define a loss function

A loss function computes a value that estimates how far away the output is from the target. The main objective is to reduce the loss function's value by changing the weight vector values through backpropagation in neural networks.

Loss value is different from model accuracy. Loss function gives us the understanding of how well a model behaves after each iteration of optimization on the training set. The accuracy of the model is calculated on the test data and shows the percentage of the right prediction.

In PyTorch, the neural network package contains various loss functions that form the building blocks of deep neural networks. In this tutorial, you will use a Classification loss function based on Define the loss function with Classification Cross-Entropy loss and an Adam Optimizer. Learning rate (lr) sets the control of how much you are adjusting the weights of our network with respect the loss gradient. You will set it as 0.001. The lower it is, the slower the training will be.

# Train the model on the training data.

To train the model, you have to loop over our data iterator, feed the inputs to the network, and optimize. PyTorch doesn’t have a dedicated library for GPU use, but you can manually define the execution device. The device will be an Nvidia GPU if exists on your machine, or your CPU if it does not.




## Run
In a terminal or command window, navigate to the top-level project directory `Create-Your-Own-Image-Classifier/` (that contains this README) and run the following command:
```bash
jupyter notebook Image-Classifier-Project.ipynb
```
on any Jupyter Notebook.
This will open the iPython Notebook software and project file in your browser.

## License
This project uses the MIT License.

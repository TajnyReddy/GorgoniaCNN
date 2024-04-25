# Convolutional Neural Network made with Gorgonia
This project implements a Convolutional Neural Network (CNN) in Go using the Gorgonia library.
![image](https://github.com/TajnyReddy/GorgoniaCNN/assets/59600478/3e5eef5c-29b8-4871-a069-e9055e8efcba)

### Architecture
* Input Layer: The input layer receives the raw pixel values of the images. Each image is resized to a standard size of 28x28 pixels to ensure uniformity.
* Convolutional Layers: The CNN consists of multiple convolutional layers, each followed by a rectified linear unit (ReLU) activation function. These layers apply learnable filters to the input images, extracting features such as edges, textures, and shapes.
* Pooling Layers: After each convolutional layer, max-pooling layers are applied to down-sample the feature maps, reducing spatial dimensions and capturing the most important information.
* Fully Connected Layers: The output from the last convolutional layer is flattened and fed into one or more fully connected layers. These layers learn to combine the extracted features from the previous layers to make predictions.
* Output Layer: The final layer of the CNN is a dense layer with a sigmoid activation function. It outputs the predicted probabilities for each class.
### Training
The CNN is trained using the RMSProp optimization algorithm, which adapts the learning rate for each parameter based on the magnitude of its gradients.
During training, the model minimizes the binary cross-entropy loss between the predicted probabilities and the actual labels of the images.
### Dropout
Dropout regularization is applied after each convolutional and fully connected layer to prevent overfitting. Dropout randomly sets a fraction of the units in the layer to zero during training, forcing the network to learn more robust features.
### Prediction
After training, the CNN can make predictions on new images by passing them through the trained model. The predicted class is determined based on the highest probability output by the network.
### Hyperparameters
Hyperparameters such as the number of epochs, batch size, dropout rates, and learning rate are adjustable to optimize the performance of the CNN.
### Customization
The architecture of the CNN can be customized by adjusting the number of convolutional layers, the size and number of filters, the pooling strategies, and the configuration of fully connected layers.

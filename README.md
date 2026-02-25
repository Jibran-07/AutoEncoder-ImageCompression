AutoEncoder Image Compression

Language Used: Python
Libraries Used: PyTorch (torch), TorchVision (torchvision), Matplotlib (matplotlib)
Dataset Used: Fashion-MNIST (contains 60,000 grayscale images)

Workflow

Defining our Model Class

The SimpleAutoencoder class inherits from nn.Module, which serves as the base class for all neural network implementations. The Module class provides methods like train() for training a model and eval() for testing.

The constructor of SimpleAutoencoder defines how the model is structured. The input_size represents the input to the model — in this case, a 28×28 Fashion-MNIST image. The hidden_size parameter represents the dimension to which the input image will be compressed and can be interpreted as the number of artificial neurons in the lower-dimensional representation.

The encoding layer performs a linear transformation from the higher-dimensional image space to a lower dimension, creating a bottleneck that forces the model to learn important features. The decoding layer performs the reverse operation.

The ReLU activation function maps each value to max(0, x), neutralizing negative values in the compressed representation. This introduces sparsity and helps the model learn more efficient feature representations.

The forward function defines the core workflow. The view() method converts 2D images into 1D vectors since linear layers expect 1D input. The -1 batch parameter instructs PyTorch to infer the batch size automatically (64 in this case). The encoded values pass through the activation function, are decoded, and then returned.

Loading Data for Training & Testing

Two separate datasets are defined: one for training and one for testing. This separation improves generalization so the model performs well on unseen data.

The ToTensor() transform scales pixel values to the range [0,1] instead of [0,255]. Large pixel values can produce large gradients and destabilize training.

Using DataLoader enables batch processing. Instead of processing images one by one, the model processes them in groups, which speeds up training. However, larger batch sizes require more memory.

Shuffling the training data prevents the model from learning order-based patterns and reduces overfitting.

Defining the Loss Function & Optimization Algorithm

The loss function measures how much the model’s predictions differ from the actual values. Mean Squared Error (MSE) computes the mean squared difference between reconstructed and original images.

The optimizer determines how the model’s weights are updated to minimize loss. The Adam optimizer adaptively adjusts learning rates for each parameter.

The learning rate (lr) controls how quickly the model learns:

Too high → may overshoot the minimum

Too low → training becomes very slow

An epoch is one full pass through the dataset. Too many epochs can cause overfitting, while too few lead to underfitting.

Training Loop

The model is set to train() mode, which enables behaviors like dropout (if present). Since this is unsupervised learning, labels are not used — the model learns to reconstruct the input.

zero_grad() clears accumulated gradients so each batch computes a fresh update.

After computing the loss, backpropagation calculates gradients, and optimizer.step() updates the weights. The epoch loss is accumulated and displayed.

Testing Loop

The model switches to evaluation mode using eval(), disabling dropout for consistent results.

The torch.no_grad() context prevents gradient computation, improving efficiency during testing. The test loss is computed over the dataset and displayed.

Displaying Reconstructed Image

The first image from the test dataset is passed through the model. The unsqueeze() method adds the required batch dimension.

Both original and reconstructed images are reshaped back to 28×28 using view() so they can be displayed with matplotlib.

Key plotting functions:

figure() sets display size

subplot() controls placement

axis('off') removes axes for cleaner visualization

Applications

Image/Audio Compression

Pattern Recognition

Database Storage Optimization

Anomaly Detection

Image Restoration

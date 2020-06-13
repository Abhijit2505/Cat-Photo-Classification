# Cat-Photo-Classification

## Model Architecture

This repository contains two models having <b>2 -</b> layers of neural network and <b>L - </b> layers of neural network respectively.
We will then compare the performance of these models, and also try out different values for  𝐿 .

Let's look at the two architectures.

### Two Layered Neural Network:

Detailed Architecture:
- The input is a (64,64,3) image which is flattened to a vector of size (12288,1). 
- The corresponding vector is then multiplied by the weight matrix.
- Then we add a bias term and take its relu.
- We then repeat the same process.
- We multiply the resulting vector by weight matrix and then add our intercept (bias). 
- Finally, we take the sigmoid of the result. If it is greater than 0.5, we classify it to be a cat.

### General Methodology:

As usual you will follow the Deep Learning methodology to build the model:

    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations:
        a. Forward propagation
        b. Compute cost function
        c. Backward propagation
        d. Update parameters (using parameters, and grads from backprop)   
    4. Use trained parameters to predict labels

#### 1. Initialize parameters and Defining Hyperparameters:

This function initializes the parameters for further use in the helper and main functions that defines the model,    
    
    Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

    Returns:
        parameters - python dictionary containing your parameters:
                W1 - weight matrix of shape (n_h, n_x)
                b1 - bias vector of shape (n_h, 1)
                W2 - weight matrix of shape (n_y, n_h)
                b2 - bias vector of shape (n_y, 1)  





















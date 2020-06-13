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

As usual you will follow the Deep Learning methodology to build the model,

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

#### 2. Forward Propagation:
Implement the linear part of a layers forward propagation,

    Arguments:
        A - activations from previous layer (or input data): (size of previous layer, number of examples)
        W - weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b - bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
        Z - the input of the activation function, also called pre-activation parameter 
        cache - a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently

Implement the forward propagation for the **LINEAR --- ACTIVATION** layer,

    Arguments:
        A_prev - activations from previous layer (or input data): (size of previous layer, number of examples)
        W - weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b - bias vector, numpy array of shape (size of the current layer, 1)
        activation - the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        A - the output of the activation function, also called the post-activation value 
        cache - a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently

#### 3. Compute Cost Function:
Implement the cost function,

    Arguments:
        AL - probability vector corresponding to your label predictions, shape (1, number of examples)
        Y - true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
        cost - cross-entropy cost

#### 4. Backward propagation:
Implement the linear portion of backward propagation for a single layer,

    Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    
Implement the backward propagation for the **LINEAR --- ACTIVATION** layer,

    Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation 
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b


#### 5. Update parameters:
Update parameters using gradient descent,
    
    Arguments:
        parameters - python dictionary containing your parameters 
        grads - python dictionary containing your gradients, output of L_model_backward
    
    Returns:
        parameters - python dictionary containing your updated parameters 
                      parameters["W" + str(l)] = ... 
                      parameters["b" + str(l)] = ...

#### 6. Predict Labels:
This function is used to predict the results of a  L-layer neural network,
    
    Arguments:
        X - data set of examples you would like to label
        parameters - parameters of the trained model

    Returns:
        p - predictions for the given dataset X
        
Currently this repository contains only the two layerd ANN, soon a L-Layred ANN having more layers and better accuracy will be added.

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

#### General Methodology:

As usual you will follow the Deep Learning methodology to build the model:

    1. Initialize parameters / Define hyperparameters
    2. Loop for num_iterations:
        a. Forward propagation
        b. Compute cost function
        c. Backward propagation
        d. Update parameters (using parameters, and grads from backprop)   
    4. Use trained parameters to predict labels

**Implementation:**
The main code in the python notebook uses some helper functions, we will be discussing those helper functions step by step,

##### 1. initialize_parameters(n_x, n_h, n_y)

    def initialize_parameters(n_x, n_h, n_y):
        np.random.seed(1)
        
        W1 = np.random.randn(n_h, n_x)*0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h)*0.01
        b2 = np.zeros((n_y, 1))

        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters
        
Argument:

    n_x -- size of the input layer   
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
Returns:

    parameters -- python dictionary containing your parameters:
       W1 -- weight matrix of shape (n_h, n_x)
       b1 -- bias vector of shape (n_h, 1)
       W2 -- weight matrix of shape (n_y, n_h) 
       b2 -- bias vector of shape (n_y, 1)

##### 2. linear_activation_forward(A_prev, W, b, activation)

    def linear_activation_forward(A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                 stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache






















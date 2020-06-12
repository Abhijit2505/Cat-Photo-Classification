# Cat-Photo-Classification

## Model Architecture

This repository contains two models having <b>2 -</b> layers of neural network and <b>L - </b> layers of neural network respectively.
We will then compare the performance of these models, and also try out different values for  𝐿 .

Let's look at the two architectures.

### Two Layered Neural Network:

<u>Detailed Architecture of figure 2</u>:
- The input is a (64,64,3) image which is flattened to a vector of size (12288,1). 
- The corresponding vector is then multiplied by the weight matrix.
- Then we add a bias term and take its relu.
- We then repeat the same process.
- We multiply the resulting vector by weight matrix and then add our intercept (bias). 
- Finally, we take the sigmoid of the result. If it is greater than 0.5, we classify it to be a cat.

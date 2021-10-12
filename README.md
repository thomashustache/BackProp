# Backpropagation 
Implementation of the back propagation algorithm from scratch on a simple regression task
## Notations:
![Screenshot](equations/notations.png)

The biases are included in the weight matrixes at each layer. Another approach would be to consider them as separate variables.
Recall that we want to minimize the loss function. Thus, we need to compute the gradient of the objective function with respect to all the parameters:

<img src="equations/obj.png" width="300"/>

Where:

<img src="equations/intermediaire1.png" width="800"/>


To ease calculations, we introduce the following error term:

<img src="equations/result1.png" width="200"/>

## Output Layer

here we have k = m.

<img src="equations/last_layer.png" width="200"/>

If the objective is the MSE loss, then the right term is easy to compute.

## Hidden Layers

here we have k < m:

<img src="equations/hidden.png" width="300"/>

## Application on a simple regression task:

Estimating the sinus function with a simple MultiLayer perceptron.
<p float="middle">
  <img src="losses.png" width="350" />
  <img src="estimations.png" width="350" /> 
</p>


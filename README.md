# poker-nn
A small NN that takes a poker hand as input and outputs the type of hand (1-pair, 2-pair, straight, flush etc). The idea was to code up a NN from scratch and see it working.

Utkarsh Kumar, 130050022

To make the whole code more modular, I have developed a class and implemented the rest as member functions:
	- __init__ : initialize the neural network with layer sizes as specified
	- feedforward: runs the neural network on the input and returns output
	- train: wrapper for batch_update and evaluate error
	- batch_update: batches the training data and calls backprop as required
	- backprop: runs backpropagation, computing derivatives and updating weights and bias parameters
	- evaluate: computed the number of points got right by the NN

Rest of the functions are helper functions for input formatting to one-hot etc.
The network takes the input as 85-bit one-hot input, has one hidden layer of 100 neurons and uses sigmoid activators everywhere.
This framework after just 50 iterations reaches 94% accuracy on training data and 92% accuracy on the testing data.
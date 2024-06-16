# NeuraFlow

A library to easily create neural networks in NodeJS

## Installation

You can install NeuraFlow using npm:

```bash
npm install https://github.com/CaraPloof/neuraflow
```

## Usage

### `NeuralNetwork` class

The `NeuralNetwork` class allows you to create and train a neural network with customizable parameters:

- Constructor Parameters:
    - `inputNodes`: Number of input nodes.
    - `hiddenNodes`: Number of nodes in the hidden layer.
    - `outputNodes`: Number of output nodes.
    - `learningRate`: Learning rate for training.
    - `epochs`: Number of training epochs.
- Methods:
    - `train(trainingInputs, trainingOutputs)`: Train the neural network using provided training data.
    - `predict(input)`: Predict outputs for a given input after training.
    - `export(path)`: Save the network's weights and biases to a specified file path.
    - `import(path)`: Load the network's weights and biases from a specified file path.

## Example
See the provided example in examples/xor.js to understand how to use NeuraFlow to solve the XOR problem.

## Credits
Project initiated by DIDELOT Tim Aka Fufly / CaraPloof. 

## License
[MIT](https://github.com/CaraPloof/neuraflow/blob/main/LICENSE)
const { NeuralNetwork } = require("../");

const inputNodes = 2;
const hiddenNodes = 5;
const outputNodes = 1;
const learningRate = 0.5;
const epochs = 10000;

const nn = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate, epochs);
// After using the export method
nn.import('xor_pretrained.json'); // Now nn has the pretrained weights and biases

const trainingInputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
];

const trainingOutputs = [
    [0],
    [1],
    [1],
    [0]
];

trainingInputs.forEach((input, index) => {
    const output = nn.predict(input);
    print(`Input: ${input} | Output: ${output[0]} | Expected: ${trainingOutputs[index][0]}`);
});
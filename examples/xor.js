const { NeuralNetwork } = require("../");

const inputNodes = 2;
const hiddenNodes = 5;
const outputNodes = 1;
const learningRate = 0.5;
const epochs = 10000;

const nn = new NeuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate, epochs);

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

nn.train(trainingInputs, trainingOutputs);

trainingInputs.forEach((input, index) => {
    const output = nn.predict(input);
    print(`Input: ${input} | Output: ${output[0]} | Expected: ${trainingOutputs[index][0]}`);
});
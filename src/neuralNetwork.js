const fs = require('fs');
const { SingleBar } = require('cli-progress');
const { shuffle } = require('./common');
const { sigmoid, dSigmoid } = require('./math');

class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate, epochs) {
        this.inputsNum = inputNodes;
        this.hiddenNodesNum = hiddenNodes;
        this.outputsNum = outputNodes;
        this.lr = learningRate;
        this.epochsNum = epochs;
        this.initWeightsAndBiases();
    }

    initWeightsAndBiases() {
        this.hiddenLayer = new Array(this.hiddenNodesNum).fill(0);
        this.outputLayer = new Array(this.outputsNum).fill(0);

        this.hiddenLayerBias = new Array(this.hiddenNodesNum).fill(0).map(() => Math.random() - 0.5);
        this.outputLayerBias = new Array(this.outputsNum).fill(0).map(() => Math.random() - 0.5);

        this.hiddenWeights = Array.from({ length: this.inputsNum }, () =>
            new Array(this.hiddenNodesNum).fill(0).map(() => Math.random() - 0.5)
        );
        this.outputWeights = Array.from({ length: this.hiddenNodesNum }, () =>
            new Array(this.outputsNum).fill(0).map(() => Math.random() - 0.5)
        );
    }


    train(trainingInputs, trainingOutputs) {
        const trainingSetOrder = Array.from(Array(trainingInputs.length).keys());

        let processedEpochs = 0;
        const progressBar = new SingleBar({
            format: 'Training Progress [{bar}] {percentage}% | ETA: {eta}s | {value}/{total} Epochs',
            barCompleteChar: '\u2588',
            barIncompleteChar: '\u2591',
            hideCursor: true
        });

        progressBar.start(this.epochsNum, 0);

        for (let e = 0; e < this.epochsNum; ++e) {
            shuffle(trainingSetOrder, trainingInputs.length);
            for (let x = 0; x < trainingInputs.length; ++x) {
                const i = trainingSetOrder[x];
                this.forward(trainingInputs[i]);
                this.backward(trainingInputs[i], trainingOutputs[i]);
            }

            processedEpochs++;
            progressBar.update(processedEpochs, { total: this.epochsNum });
        }

        progressBar.stop();
    }

    forward(input) {
        // Compute hidden layer activation
        for (let j = 0; j < this.hiddenNodesNum; ++j) {
            let activation = this.hiddenLayerBias[j];
            for (let k = 0; k < this.inputsNum; ++k) {
                activation += input[k] * this.hiddenWeights[k][j];
            }
            this.hiddenLayer[j] = sigmoid(activation);
        }

        // Compute output layer activation
        for (let j = 0; j < this.outputsNum; ++j) {
            let activation = this.outputLayerBias[j];
            for (let k = 0; k < this.hiddenNodesNum; ++k) {
                activation += this.hiddenLayer[k] * this.outputWeights[k][j];
            }
            this.outputLayer[j] = sigmoid(activation);
        }
    }

    backward(input, target) {
        // Compute change in output weights
        const deltaOutput = new Array(this.outputsNum);
        for (let j = 0; j < this.outputsNum; ++j) {
            const error = (target[j] - this.outputLayer[j]);
            deltaOutput[j] = error * dSigmoid(this.outputLayer[j]);
        }

        // Compute change in hidden weights
        const deltaHidden = new Array(this.hiddenNodesNum);
        for (let j = 0; j < this.hiddenNodesNum; ++j) {
            let error = 0.0;
            for (let k = 0; k < this.outputsNum; ++k) {
                error += deltaOutput[k] * this.outputWeights[j][k];
            }
            deltaHidden[j] = error * dSigmoid(this.hiddenLayer[j]);
        }

        // Apply change in output weights
        for (let j = 0; j < this.outputsNum; ++j) {
            this.outputLayerBias[j] += deltaOutput[j] * this.lr;
            for (let k = 0; k < this.hiddenNodesNum; ++k) {
                this.outputWeights[k][j] += this.hiddenLayer[k] * deltaOutput[j] * this.lr;
            }
        }

        // Apply change in hidden weights
        for (let j = 0; j < this.hiddenNodesNum; ++j) {
            this.hiddenLayerBias[j] += deltaHidden[j] * this.lr;
            for (let k = 0; k < this.inputsNum; ++k) {
                this.hiddenWeights[k][j] += input[k] * deltaHidden[j] * this.lr;
            }
        }
    }

    predict(input) {
        this.forward(input);
        return this.outputLayer;
    }

    export(path) {
        const data = {
            hiddenLayerBias: this.hiddenLayerBias,
            outputLayerBias: this.outputLayerBias,
            hiddenWeights: this.hiddenWeights,
            outputWeights: this.outputWeights
        };
        fs.writeFileSync(path, JSON.stringify(data));
    }

    import(path) {
        const data = JSON.parse(fs.readFileSync(path, 'utf8'));
        this.hiddenLayerBias = data.hiddenLayerBias;
        this.outputLayerBias = data.outputLayerBias;
        this.hiddenWeights = data.hiddenWeights;
        this.outputWeights = data.outputWeights;
    }
}

exports.NeuralNetwork = NeuralNetwork;
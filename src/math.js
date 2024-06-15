const sigmoid = (x) => { return 1 / (1 + Math.exp(-x)); };
const dSigmoid = (x) => { return x * (1 - x); };

module.exports = {
    sigmoid,
    dSigmoid
};
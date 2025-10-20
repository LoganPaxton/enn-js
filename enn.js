import { readFileSync, writeFileSync } from "fs";

class Layer {
    constructor(input_size, output_size, activation = "sigmoid") {
        this.weights = Array.from({ length: output_size }, () =>
            Array.from({ length: input_size }, () => Math.random() * 2 - 1)
        );
        this.bias = Array.from({ length: output_size }, () => Math.random() * 2 - 1);
        this.activation = activation;
    }

    activate(x) {
        switch (this.activation) {
            case "relu": return Math.max(0, x);
            case "sigmoid": return 1 / (1 + Math.exp(-x));
            default: return x; // linear
        }
    }

    derivative(outputValue) {
        switch (this.activation) {
            case "relu": return outputValue > 0 ? 1 : 0;
            case "sigmoid": return outputValue * (1 - outputValue);
            default: return 1;
        }
    }
}

export class Brain {
    constructor(input_nodes, hidden_nodes, output_nodes, activation = "sigmoid") {
        this.layers = [
            new Layer(input_nodes, hidden_nodes, activation),
            new Layer(hidden_nodes, output_nodes, activation),
        ];
    }

    static dot(a, b) {
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    // Feedforward and record activations
    feedForward(inputs) {
        const activations = [inputs];
        let outputs = inputs;

        for (let layer of this.layers) {
            const next_outputs = [];

            for (let i = 0; i < layer.weights.length; i++) {
                const weighted_sum = Brain.dot(outputs, layer.weights[i]) + layer.bias[i];
                next_outputs.push(layer.activate(weighted_sum));
            }

            outputs = next_outputs;
            activations.push(outputs);
        }

        this._lastActivations = activations;
        return outputs;
    }

    backpropagation(sample, lr) {
        if (!this._lastActivations) {
            this.feedForward(sample.input);
        }

        const activations = this._lastActivations;
        let errors = activations.at(-1).map((out, i) => sample.output[i] - out);

        for (let l = this.layers.length - 1; l >= 0; l--) {
            const layer = this.layers[l];
            const outputs = activations[l + 1];
            const prev_outputs = activations[l];

            // Gradient = error * derivative(output)
            const gradients = outputs.map((out, i) => errors[i] * layer.derivative(out));

            // Update weights and biases
            for (let i = 0; i < layer.weights.length; i++) {
                for (let j = 0; j < layer.weights[i].length; j++) {
                    layer.weights[i][j] += lr * gradients[i] * prev_outputs[j];
                }
                layer.bias[i] += lr * gradients[i];
            }

            // Calculate errors for previous layer
            if (l > 0) {
                const new_errors = new Array(layer.weights[0].length).fill(0);
                for (let i = 0; i < layer.weights.length; i++) {
                    for (let j = 0; j < layer.weights[i].length; j++) {
                        new_errors[j] += gradients[i] * layer.weights[i][j];
                    }
                }
                errors = new_errors;
            }
        }
    }

    train(lr, training_data, epochs = 10000) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let total_loss = 0;

            for (let sample of training_data) {
                const predictions = this.feedForward(sample.input);
                this.backpropagation(sample, lr);

                const loss = predictions.reduce(
                    (acc, p, i) => acc + (sample.output[i] - p) ** 2,
                    0
                );
                total_loss += loss;
            }

            if (epoch % 1000 === 0) {
                console.log(
                    `Epoch ${epoch}: Loss = ${(total_loss / training_data.length).toFixed(6)}`
                );
            }
        }
    }

    save(file = "weights.json") {
        const data = JSON.stringify(this.layers, null, 2);
        writeFileSync(file, data);
    }

    load(file = "weights.json") {
        const data = JSON.parse(readFileSync(file));
        this.layers = data.map(l => Object.assign(new Layer(0, 0), l));
    }
}

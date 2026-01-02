import { readFileSync, writeFileSync } from "fs";

class Layer {
    constructor(input_size, output_size, activation = "sigmoid") {
        const scale = Math.sqrt(2 / input_size)
        this.weights = Array.from({ length: output_size }, () =>
            Array.from({ length: input_size }, () => (Math.random() * 2 - 1) * scale)
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

    derivative(output) {
        switch (this.activation) {
            case "relu": return output > 0 ? 1 : 0;
            case "sigmoid": return output * (1 - output);
            default: return 1;
        }
    }
}

export class Brain {
    constructor(
        input_nodes,
        hidden_nodes,
        output_nodes,
        hidden_activation = "sigmoid",
        output_activation = "softmax"
    ) {
        this.layers = [
            new Layer(input_nodes, hidden_nodes, hidden_activation),
            new Layer(hidden_nodes, output_nodes, output_activation),
        ];
    }

    static dot(a, b) {
        return a.reduce((sum, v, i) => sum + v * b[i], 0);
    }

    static softmax(vec) {
        const max = Math.max(...vec);
        const exps = vec.map(v => Math.exp(Math.min(v - max, 50)));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    feedForward(inputs) {
        const activations = [inputs];
        let outputs = inputs;

        for (const layer of this.layers) {
            const z = layer.weights.map(
                (w, i) => Brain.dot(outputs, w) + layer.bias[i]
            );

            if (layer.activation === "softmax") {
                outputs = Brain.softmax(z);
            } else {
                outputs = z.map(v => layer.activate(v));
            }

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
        const predictions = activations.at(-1);
        const outputLayer = this.layers.at(-1);

        // Output error
        let errors;
        if (outputLayer.activation === "softmax") {
            // Softmax + cross-entropy
            errors = predictions.map((p, i) => p - sample.output[i]);
        } else {
            errors = predictions.map((p, i) => sample.output[i] - p);
        }

        for (let l = this.layers.length - 1; l >= 0; l--) {
            const layer = this.layers[l];
            const outputs = activations[l + 1];
            const prev = activations[l];

            const gradients = outputs.map((out, i) => {
                if (l === this.layers.length - 1 && layer.activation === "softmax") {
                    return errors[i]; // already dL/dz
                }
                return errors[i] * layer.derivative(out);
            });

            // Update weights & biases
            for (let i = 0; i < layer.weights.length; i++) {
                for (let j = 0; j < layer.weights[i].length; j++) {
                    layer.weights[i][j] -= lr * gradients[i] * prev[j];
                }
                layer.bias[i] -= lr * gradients[i];
            }

            // Propagate error backward
            if (l > 0) {
                const newErrors = new Array(layer.weights[0].length).fill(0);
                for (let i = 0; i < layer.weights.length; i++) {
                    for (let j = 0; j < layer.weights[i].length; j++) {
                        newErrors[j] += gradients[i] * layer.weights[i][j];
                    }
                }
                errors = newErrors;
            }
        }
    }

    train(lr, training_data, epochs = 10000) {
        training_data = training_data // Shuffle training data
            .map(v => ({ v, r: Math.random() }))
            .sort((a, b) => a.r - b.r)
            .map(x => x.v);
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            let total_loss = 0;

            for (const sample of training_data) {
                const predictions = this.feedForward(sample.input);
                this.backpropagation(sample, lr);

                const eps = 1e-9;
                const loss = predictions.reduce(
                    (acc, p, i) => acc - sample.output[i] * Math.log(Math.min(Math.max(p, eps), 1 - eps)),
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

    predict(input) {
        const output = this.feedForward(input);
        return output.indexOf(Math.max(...output));
    }

    save(file = "weights.json") {
        writeFileSync(file, JSON.stringify(this.layers, null, 2));
    }

    load(file = "weights.json") {
        const data = JSON.parse(readFileSync(file));
        this.layers = data.map(l => Object.assign(new Layer(0, 0), l));
    }
}

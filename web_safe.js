class Brain {
    constructor(input_nodes, hidden_nodes, output_nodes) {
        this.hidden_weights = [];
        this.output_weights = [];

        this.hidden_bias = [];
        this.output_bias = [];

        
        // WEIGHTS
        // Assign random values for hidden weights
        for (let i = 0; i < hidden_nodes; i++) {
            this.hidden_weights[i] = [];
            for (let j = 0; j < input_nodes; j++) {
                this.hidden_weights[i][j] = Math.random() * 2 - 1;
            }
        }

        // Assign random values for output weights
        for (let i = 0; i < output_nodes; i++) {
            this.output_weights[i] = [];
            for (let j = 0; j < hidden_nodes; j++) {
                this.output_weights[i][j] = Math.random() * 2 - 1;
            }
        }

        // BIAS
        // Assign random values for hidden bias
        for (let i = 0; i != hidden_nodes; i++) {
            let randfloat = Math.random() * 2 - 1
            this.hidden_bias[i] = randfloat;
        }

        // Assign random values for output bias
        for (let i = 0; i != output_nodes; i++) {
            let randfloat = Math.random() * 2 - 1
            this.output_bias[i] = randfloat;
        }
    }

    // MATH FUNCTIONS
    // Sigmoid Formula: 1 / (1 + e^-x)
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Calculating the hidden layer output
    forwardHidden(inputs) {
        let hidden_layer_outputs = [];

        for (let i = 0; i < this.hidden_weights.length; i++) {
            let sum = inputs.reduce((acc, val, idx) => acc + val * this.hidden_weights[i][idx], this.hidden_bias[i]);
            hidden_layer_outputs.push(this.sigmoid(sum))
        }
        this.hidden_layer_outputs = hidden_layer_outputs;
    }

    feedforward(inputs) {
        this.forwardHidden(inputs);
    
        let output = [];
        for (let i = 0; i < this.output_weights.length; i++) {
            let sum = this.hidden_layer_outputs.reduce(
                (acc, val, idx) => acc + val * this.output_weights[i][idx],
                this.output_bias[i]
            );
            output.push(this.sigmoid(sum));
        }
    
        return output;
    }
    

    calculate_output() {
        let output_layer_outputs = [];

        for (let i = 0; i < this.output_weights.length; i++) {
            let sum = this.hidden_layer_outputs.reduce((acc, val, idx) => acc + val * this.output_weights[i][idx], this.output_bias[i]);
            output_layer_outputs.push(this.sigmoid(sum));
        }
    this.output_layer_outputs = output_layer_outputs;
    }

    backpropagation(sample) {
        let targets = sample.output;
        let output_error = [];
        let hidden_error = [];

        // Calculate the output error based on the targets
        for (let i = 0; i < this.output_layer_outputs.length; i++) {
            output_error[i] = targets[i] - this.output_layer_outputs[i];
        }

        // Calculate the hidden error based on the output layer
        for (let j = 0; j < this.hidden_weights.length; j++) {
            let sum = 0;
            for (let i = 0; i < this.output_weights.length; i++) {
                sum += output_error[i] * this.output_weights[i][j];
            }
            hidden_error[j] = sum;
        }

        // Calculate the hidden graident based on the hidden error, and sigmoid derivative
        let hidden_gradient = [];

        for (let i = 0; i < hidden_error.length; i++) {
            hidden_gradient[i] = hidden_error[i] * (this.hidden_layer_outputs[i] * (1 - this.hidden_layer_outputs[i]))
        }

        // Calculate the output graident based on the ouput error, and sigmoid derivative
        let output_gradient = [];

        for (let i = 0; i < output_error.length; i++) {
            output_gradient[i] = output_error[i] * (this.output_layer_outputs[i] * (1 - this.output_layer_outputs[i]))
        }

        // Adjust hidden weights based on hidden_gradient
        for (let j = 0; j < this.hidden_weights.length; j++) {
            for (let k = 0; k < this.hidden_weights[j].length; k++) {
                let delta = this.learning_rate * hidden_gradient[j] * sample.input[k];
                this.hidden_weights[j][k] += delta;
            }
        }
        
        // Adjust output weights based on output_gradient
        for (let i = 0; i < this.output_weights.length; i++) {
            for (let j = 0; j < this.output_weights[i].length; j++) {
                let delta = this.learning_rate * output_gradient[i] * this.hidden_layer_outputs[j];
                this.output_weights[i][j] += delta;
            }
        }    
    }

    train(lr, training_data, max_epoch) {
        this.training_data = training_data;
        this.learning_rate = lr;

        
        for (let epoch = 0; epoch < max_epoch; epoch++) {
            training_data.forEach(sample => {
                this.forwardHidden(sample.input);
                this.calculate_output();
                this.backpropagation(sample);
            });

        }

    }
}
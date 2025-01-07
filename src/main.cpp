#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include "Neuron.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"

using namespace kl;

void testXOR() {
    // Define XOR topology
    std::vector<uint> topology = {2, 4, 1}; // 2 inputs, 3 hidden neurons, 1 output
    double learningRate = 0.01;

    // Create neural network
    NeuralNetwork nn(topology, learningRate);

    // XOR inputs and outputs
    Eigen::MatrixXd inputs(4, 2);
    inputs << 0, 0,
              0, 1,
              1, 0,
              1, 1;

    Eigen::MatrixXd targets(4, 1);
    targets << 0,
               1,
               1,
               0;

    // Train network
    int epochs = 6000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (Eigen::Index i = 0; i < inputs.rows(); ++i) {
            nn.forwardPropagation(inputs.row(i));
            nn.backPropagation(targets.row(i));
            nn.updateWeightsAndBiases();
            totalError += nn.calculateError(targets.row(i));
        }

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ": Error = " << totalError / inputs.rows() << std::endl;
        }
    }

    // Test network
    std::cout << "XOR Test Results:\n";
    for (Eigen::Index i = 0; i < inputs.rows(); ++i) {
        Eigen::VectorXd output = nn.predict(inputs.row(i));
        std::cout << "Input: " << inputs.row(i)
                  << ", Predicted: " << output[0]
                  << ", Target: " << targets(i, 0) << std::endl;
    }
}

void testSineFunction() {
    // Define topology for sine function approximation
    std::vector<uint> topology = {1, 6, 1}; // 1 input, 2 hidden neurons, 1 output
    double learningRate = 0.01;

    // Create neural network
    NeuralNetwork nn(topology, learningRate);

    // Generate training data for sine function
    int samples = 50;
    Eigen::MatrixXd inputs(samples, 1);
    Eigen::MatrixXd targets(samples, 1);
    for (int i = 0; i < samples; ++i) {
        double x = -M_PI + i * (2 * M_PI / samples);
        inputs(i, 0) = x;
        targets(i, 0) = std::sin(x);
    }

    // Train network
    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (Eigen::Index i = 0; i < inputs.rows(); ++i) {
            nn.forwardPropagation(inputs.row(i));
            nn.backPropagation(targets.row(i));
            nn.updateWeightsAndBiases();
            totalError += nn.calculateError(targets.row(i));
        }

        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << ": Error = " << totalError / inputs.rows() << std::endl;
        }
    }

    // Test network
    std::cout << "Sine Function Approximation Results:\n";
    for (int i = 0; i < samples; i += 5) {
        Eigen::VectorXd output = nn.predict(inputs.row(i));
        std::cout << "Input: " << inputs(i, 0)
                  << ", Predicted: " << output[0]
                  << ", Target: " << targets(i, 0) << std::endl;
    }
}

int main() {
    std::cout << "Running XOR Test...\n";
    testXOR();

    std::cout << "\nRunning Sine Function Test...\n";
    testSineFunction();

    return 0;
}

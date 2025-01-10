/**
 * @file main.cpp
 * @brief Entry point for the neural network application.
 * 
 * This file contains the main function to test the neural network implementation
 * using the XOR problem as an example.
 */

#include <Eigen/Dense>

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

#include "Neuron.hpp"
#include "Layer.hpp"
#include "NeuralNetwork.hpp"
#include "utils.hpp"
#include "main.hpp"

using kl::NeuralNetwork;

void testXOR() {
    // Define XOR topology
    std::vector<uint> topology = {2, 4, 1}; // 2 inputs, 4 hidden neurons, 1 output
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
    std::vector<uint> topology = {1, 6, 1}; // 1 input, 6 hidden neurons, 1 output
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

void testIris() {
    // Load the Iris dataset
    Eigen::MatrixXd irisData = readCSV("../data/iris.csv");
    Eigen::MatrixXd irisTargets = readCSV("../data/iris_out.csv");

    // Normalize the input data
    Eigen::MatrixXd normalizedData = normalizeMatrix(irisData);

    // Define topology for the neural network
    std::vector<uint> topology = {4, 5, 4, 3}; // 4 input, 5 hidden, 4 hidden, 3 output
    double learningRate = 0.01;

    // Create the neural network
    NeuralNetwork nn(topology, learningRate);

    // Training parameters
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalError = 0.0;
        for (Eigen::Index i = 0; i < normalizedData.rows(); ++i) {
            Eigen::VectorXd input = normalizedData.row(i);
            Eigen::VectorXd target = irisTargets.row(i);

            nn.forwardPropagation(input);
            nn.backPropagation(target);
            nn.updateWeightsAndBiases();

            totalError += nn.calculateError(target);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ": Total Error = " << totalError / normalizedData.rows() << std::endl;
        }
    }

    // Custom test case
    std::cout << "\nTesting custom input:\n";
    Eigen::VectorXd customInput(4);
    customInput << 7.9, 3.8, 6.4, 2.0; // Example Iris Virginica flower measurements
    Eigen::VectorXd normalizedCustomInput = normalizeInput(customInput, irisData);
    Eigen::VectorXd customOutput = nn.predict(normalizedCustomInput);

    std::cout << "Custom Input: " << customInput.transpose() << "\n"
            << "Predicted Output: " << customOutput.transpose() << "\n";
}

int main() {
    std::cout << "Running XOR Test...\n";
    testXOR();

    std::cout << "\nRunning Sine Function Test...\n";
    testSineFunction();

    std::cout << "\nRunning Iris Dataset Test...\n";
    testIris();

    return 0;
}

/**
 * @file NeuralNetwork.hpp
 * @brief Defines the NeuralNetwork class for managing layers, training, and predictions.
 */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "Layer.hpp"

namespace kl {

/**
 * @class NeuralNetwork
 * @brief Represents a neural network with multiple layers.
 */
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> m_layers; ///< Layers of the neural network.
    std::vector<uint> m_topology;                ///< Topology of the neural network.
    double m_learningRate;                       ///< Learning rate for training.

    /**
     * @brief Initializes weights and biases for the network.
     */
    void initializeWeightsAndBiases();

public:
    /**
     * @brief Constructs a NeuralNetwork with the specified topology and learning rate.
     * @param topology A vector representing the number of neurons in each layer.
     * @param learningRate The learning rate for training.
     */
    NeuralNetwork(const std::vector<uint>& topology, double learningRate);

    /**
     * @brief Performs forward propagation through the network.
     * @param input The input vector.
     */
    void forwardPropagation(const Eigen::VectorXd& input);

    /**
     * @brief Performs backpropagation to calculate gradients.
     * @param target The target output vector.
     */
    void backPropagation(const Eigen::VectorXd& target);

    /**
     * @brief Updates weights and biases using gradients and learning rate.
     */
    void updateWeightsAndBiases();

    /**
     * @brief Predicts the output for a given input vector.
     * @param input The input vector to predict.
     * @return Eigen::VectorXd The predicted output vector.
     */
    Eigen::VectorXd predict(const Eigen::VectorXd& input);

    /**
     * @brief Calculates the total error using mean squared error.
     * @param target The target output vector.
     * @return double The total error.
     */
    double calculateError(const Eigen::VectorXd& target) const;
};

} // namespace kl

#endif // NEURAL_NETWORK_HPP

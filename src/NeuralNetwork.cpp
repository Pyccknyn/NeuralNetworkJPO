/**
 * @file NeuralNetwork.cpp
 * @brief Implements the NeuralNetwork class for managing layers, training, and predictions.
 */
#include "NeuralNetwork.hpp"
#include <random>

namespace kl {

/**
 * @brief Constructs a NeuralNetwork with the specified topology and learning rate.
 * @param topology A vector representing the number of neurons in each layer.
 * @param learningRate The learning rate for training.
 */
NeuralNetwork::NeuralNetwork(const std::vector<uint>& topology, double learningRate)
    : m_topology(topology), m_learningRate(learningRate) {
    m_layers.push_back(std::make_unique<InputLayer>(topology[0]));
    for (size_t i = 1; i < topology.size() - 1; ++i) {
        m_layers.push_back(std::make_unique<HiddenLayer>(topology[i]));
    }
    m_layers.push_back(std::make_unique<OutputLayer>(topology.back()));
    
    for (size_t i = 1; i < m_layers.size(); ++i) {
        m_layers[i]->m_previousLayer = m_layers[i - 1].get(); // Link previous layer
    }
    for (size_t i = 0; i < m_layers.size() - 1; ++i) {
        m_layers[i]->m_nextLayer = m_layers[i + 1].get();     // Link next layer
    }

    initializeWeightsAndBiases();
}

/**
 * @brief Initializes the weights and biases for all neurons in the network.
 * Weights and biases are initialized using a normal distribution.
 */
void NeuralNetwork::initializeWeightsAndBiases() {
    std::random_device rd;
    std::mt19937 gen(rd());
    for (size_t i = 1; i < m_layers.size(); ++i) {
        auto& prevNeurons = m_layers[i - 1]->getNeurons();
        double stddev = std::sqrt(2.0 / (prevNeurons.size() + m_layers[i]->getNeurons().size()));
        std::normal_distribution<> dis(0.0, stddev);

        for (auto& neuron : m_layers[i]->getNeurons()) {
            neuron.setBias(dis(gen)); // Initialize bias
            std::vector<double> weights(prevNeurons.size());
            for (auto& weight : weights) {
                weight = dis(gen); // Initialize weights
            }
            neuron.setWeights(weights);
        }
    }
}

/**
 * @brief Performs forward propagation through the network.
 * @param input The input vector to be passed through the network.
 */
void NeuralNetwork::forwardPropagation(const Eigen::VectorXd& input) {
    dynamic_cast<InputLayer*>(m_layers[0].get())->forwardParam(input);
    for (size_t i = 1; i < m_layers.size(); ++i) {
        m_layers[i]->forward();
    }
}

/**
 * @brief Performs backpropagation to calculate gradients for training.
 * @param target The target output vector used for error calculation.
 */
void NeuralNetwork::backPropagation(const Eigen::VectorXd& target) {
    dynamic_cast<OutputLayer*>(m_layers.back().get())->backwardParam(target);
    for (size_t i = m_layers.size() - 2; i > 0; --i) {
        m_layers[i]->backward();
    }
}

/**
 * @brief Updates weights and biases using the calculated gradients and learning rate.
 */
void NeuralNetwork::updateWeightsAndBiases() {
    for (size_t i = 1; i < m_layers.size(); ++i) {
        auto& prevNeurons = m_layers[i - 1]->getNeurons();
        for (auto& neuron : m_layers[i]->getNeurons()) {
            auto weights = neuron.getWeights();
            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] += m_learningRate * neuron.getGradient() * prevNeurons[j].getActivation();
            }
            neuron.setWeights(weights);
            neuron.setBias(neuron.getBias() + m_learningRate * neuron.getGradient());
        }
    }
}

/**
 * @brief Predicts the output for a given input vector.
 * @param input The input vector to predict the output for.
 * @return Eigen::VectorXd The predicted output vector.
 */
Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
    forwardPropagation(input);
    Eigen::VectorXd output(m_layers.back()->getNeurons().size());
    for (Eigen::Index i = 0; i < output.size(); ++i) {
        output[i] = m_layers.back()->getNeurons()[i].getActivation();
    }
    return output;
}

/**
 * @brief Calculates the total error of the network using mean squared error.
 * @param target The target output vector.
 * @return double total error value.
 */
double NeuralNetwork::calculateError(const Eigen::VectorXd& target) const {
    const auto& outputNeurons = m_layers.back()->getNeurons();
    double totalError = 0.0;
    for (Eigen::Index i = 0; i < target.size(); ++i) {
        double activation = outputNeurons[i].getActivation();
        totalError += 0.5 * std::pow(target[i] - activation, 2); // Mean squared error
    }
    return totalError;
}

} // namespace kl

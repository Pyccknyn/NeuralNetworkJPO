/**
 * @file Layer.cpp
 * @brief Implements the Layer base class and derived classes InputLayer, HiddenLayer, OutputLayer.
 */
#include "Layer.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace kl {

/**
 * @brief Gets a constant reference to the neurons in the layer.
 * @return A constant reference to the vector of neurons.
 */
const std::vector<Neuron>& Layer::getNeurons() const {return m_neurons;}

/**
 * @brief Gets a mutable reference to the neurons in the layer.
 * @return A reference to the vector of neurons.
 */
std::vector<Neuron>& Layer::getNeurons() {return m_neurons;}

/**
 * @brief Sets the previous layer in the network.
 * @param previousLayer Pointer to the previous layer.
 */
void Layer::setPreviousLayer(Layer* previousLayer) {m_previousLayer = previousLayer;}

/**
 * @brief Sets the next layer in the network.
 * @param nextLayer Pointer to the next layer.
 */
void Layer::setNextLayer(Layer* nextLayer) {m_nextLayer = nextLayer;}

/**
 * @brief Constructs an InputLayer with the specified number of neurons.
 * @param numNeurons Number of neurons in the layer.
 */
InputLayer::InputLayer(int numNeurons) {
    m_neurons.resize(numNeurons);
}

/**
 * @brief Performs a forward pass with input values.
 * @param input Vector of input values to the layer.
 */
void InputLayer::forwardParam(const Eigen::VectorXd& input) {
    for (size_t i = 0; i < m_neurons.size(); ++i) {
        m_neurons[i].setValue(input[i]);
        m_neurons[i].setActivation(input[i]);
    }
}

/**
 * @brief Performs a forward pass (not used in InputLayer).
 */
void InputLayer::forward() {
    // This method remains empty because `InputLayer::forward(const Eigen::VectorXd&)` is used
}

/**
 * @brief Performs a backward pass (not required for InputLayer).
 */
void InputLayer::backward() {
    // InputLayer does not require backpropagation
}

/**
 * @brief Constructs a HiddenLayer with the specified number of neurons.
 * @param numNeurons Number of neurons in the layer.
 */
HiddenLayer::HiddenLayer(int numNeurons) {
    m_neurons.resize(numNeurons);
}

/**
 * @brief Performs a forward pass through the hidden layer.
 */
void HiddenLayer::forward() {
    for (auto& neuron : m_neurons) {
        double weightedSum = 0.0;
        for (size_t i = 0; i < m_previousLayer->getNeurons().size(); ++i) {
            weightedSum += m_previousLayer->getNeurons()[i].getActivation() * neuron.getWeights()[i];
        }
        weightedSum += neuron.getBias();
        neuron.setActivation(Neuron::tanhActivation(weightedSum)); // Use tanh
    }
}

/**
 * @brief Performs a backward pass through the hidden layer.
 */
void HiddenLayer::backward() {
    for (size_t i = 0; i < m_neurons.size(); ++i) {
        double downstreamGradientSum = 0.0;
        for (const auto& downstreamNeuron : m_nextLayer->getNeurons()) {
            downstreamGradientSum += downstreamNeuron.getWeights()[i] * downstreamNeuron.getGradient();
        }
        double activation = m_neurons[i].getActivation();
        double gradient = downstreamGradientSum * Neuron::tanhDerivative(activation); // Use tanh derivative
        m_neurons[i].setGradient(gradient);
    }
}

/**
 * @brief Constructs an OutputLayer with the specified number of neurons.
 * @param numNeurons Number of neurons in the layer.
 */
OutputLayer::OutputLayer(int numNeurons) {
    m_neurons.resize(numNeurons);
}

/**
 * @brief Performs a forward pass through the output layer.
 */
void OutputLayer::forward() {
    for (auto& neuron : m_neurons) {
        double weightedSum = 0.0;
        for (size_t i = 0; i < m_previousLayer->getNeurons().size(); ++i) {
            weightedSum += m_previousLayer->getNeurons()[i].getActivation() * neuron.getWeights()[i];
        }
        weightedSum += neuron.getBias();
        neuron.setActivation(Neuron::tanhActivation(weightedSum)); // Use tanh
    }
}

/**
 * @brief Performs a backward pass through the output layer using target values.
 * @param target Vector of target values for backpropagation.
 */
void OutputLayer::backwardParam(const Eigen::VectorXd& target) {
    for (size_t i = 0; i < m_neurons.size(); ++i) {
        double activation = m_neurons[i].getActivation();
        double error = target[i] - activation;
        double gradient = error * Neuron::tanhDerivative(activation); // Use tanh derivative
        m_neurons[i].setGradient(gradient);
    }
}

/**
 * @brief Performs a backward pass (not used in this overload).
 */
void OutputLayer::backward() {
    // This method remains empty because `OutputLayer::backward(const Eigen::VectorXd&)` is used
}

} // namespace kl
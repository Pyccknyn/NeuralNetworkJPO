/**
 * @file Layer.cpp
 * @brief Implements the Layer base class and derived classes InputLayer, HiddenLayer, OutputLayer.
 */

#include "Layer.hpp"
#include "Neuron.hpp"
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
    }
}

/**
 * @brief Performs a forward pass (overridden from the base class).
 */
void InputLayer::forward() {
    // Implementation for forward pass in InputLayer
}

/**
 * @brief Performs a backward pass (overridden from the base class).
 */
void InputLayer::backward() {
    // Implementation for backward pass in InputLayer
}

/**
 * @brief Constructs a HiddenLayer with the specified number of neurons.
 * @param numNeurons Number of neurons in the layer.
 */
HiddenLayer::HiddenLayer(int numNeurons) {
    m_neurons.resize(numNeurons);
}

/**
 * @brief Performs a forward pass (overridden from the base class).
 */
void HiddenLayer::forward() {
    // Implementation for forward pass in HiddenLayer
}

/**
 * @brief Performs a backward pass (overridden from the base class).
 */
void HiddenLayer::backward() {
    // Implementation for backward pass in HiddenLayer
}

/**
 * @brief Constructs an OutputLayer with the specified number of neurons.
 * @param numNeurons Number of neurons in the layer.
 */
OutputLayer::OutputLayer(int numNeurons) {
    m_neurons.resize(numNeurons);
}

/**
 * @brief Performs a forward pass (overridden from the base class).
 */
void OutputLayer::forward() {
    // Implementation for forward pass in OutputLayer
}

/**
 * @brief Performs a backward pass with target values.
 * @param target Vector of target values for backpropagation.
 */
void OutputLayer::backwardParam(const Eigen::VectorXd& target) {
    // Implementation for backward pass with target values in OutputLayer
}

/**
 * @brief Performs a backward pass (overridden from the base class).
 */
void OutputLayer::backward() {
    // Implementation for backward pass in OutputLayer
}

} // namespace kl
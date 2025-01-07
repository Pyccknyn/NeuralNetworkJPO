#include "Neuron.hpp"

namespace kl {

/**
 * @brief Default constructor for the Neuron class.
 * Initializes m_value, m_bias, m_activation, and m_gradient to 0.
 * Initializes m_weights to an empty vector.
 */
Neuron::Neuron() : m_value(0.0), m_bias(0.0), m_activation(0.0), m_gradient(0.0) {}

/**
 * @brief Sets the input value of the neuron.
 * @param value The input value to set.
 */
void Neuron::setValue(double value) { m_value = value; }

/**
 * @brief Gets the input value of the neuron.
 * @return The input value of the neuron.
 */
double Neuron::getValue() const { return m_value; }

/**
 * @brief Sets the bias of the neuron.
 * @param bias The bias value to set.
 */
void Neuron::setBias(double bias) { m_bias = bias; }

/**
 * @brief Gets the bias of the neuron.
 * @return The bias of the neuron.
 */
double Neuron::getBias() const { return m_bias; }

/**
 * @brief Sets the activation value of the neuron.
 * @param activation The activation value to set.
 */
void Neuron::setActivation(double activation) { m_activation = activation; }

/**
 * @brief Gets the activation value of the neuron.
 * @return The activation value of the neuron.
 */
double Neuron::getActivation() const { return m_activation; }

/**
 * @brief Sets the gradient for backpropagation.
 * @param gradient The gradient value to set.
 */
void Neuron::setGradient(double gradient) { m_gradient = gradient; }

/**
 * @brief Gets the gradient for backpropagation.
 * @return The gradient value of the neuron.
 */
double Neuron::getGradient() const { return m_gradient; }

/**
 * @brief Sets the weights connecting to the previous layer.
 * @param weights A vector of weights to set.
 */
void Neuron::setWeights(const std::vector<double>& weights) { m_weights = weights; }

/**
 * @brief Gets the weights connecting to the previous layer.
 * @return A vector of weights of the neuron.
 */
std::vector<double> Neuron::getWeights() const { return m_weights; }

/**
 * @brief Hyperbolic tangent activation function.
 * @param x The input value.
 * @return The activated value using the tanh function.
 */
double Neuron::tanhActivation(double x) {
    return std::tanh(x); 
}

/**
 * @brief Derivative of the hyperbolic tangent activation function.
 * @param x The input value.
 * @return The derivative value of the tanh function.
 */
double Neuron::tanhDerivative(double x) {
    double tanhValue = tanhActivation(x);
    return 1.0 - (tanhValue * tanhValue); // 1 - tanh^2(x)
}

} // namespace kl

/**
* @file Neuron.hpp
* @brief Defines the Neuron class for representing individual units in the neural network.
*/
#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <cmath>

namespace kl {

/**
* @class Neuron
* @brief Represents a single neuron in the neural network.
*/
class Neuron {
private:
    double m_value;                     ///< The raw input value of the neuron.
    double m_bias;                      ///< The bias value of the neuron.
    double m_activation;                ///< The activation value of the neuron.
    double m_gradient;                  ///< The gradient for backpropagation.
    std::vector<double> m_weights;      ///< Weights connecting to the previous layer.

public:
    /**
     * @brief Default constructor for the Neuron class.
     * Initializes m_value, m_bias, m_activation, and m_gradient to 0.
     * Initializes m_weights to an empty vector.
     */
    Neuron();

    /**
     * @brief Sets the input value of the neuron.
     * @param value The input value to set.
     */
    void setValue(double value); 

    /**
     * @brief Gets the input value of the neuron.
     * @return The input value of the neuron.
     */
    double getValue() const;

    /**
     * @brief Sets the bias of the neuron.
     * @param bias The bias value to set.
     */
    void setBias(double bias); 

    /**
     * @brief Gets the bias of the neuron.
     * @return The bias of the neuron.
     */
    double getBias() const;

    /**
     * @brief Sets the activation value of the neuron.
     * @param activation The activation value to set.
     */
    void setActivation(double activation);

    /**
     * @brief Gets the activation value of the neuron.
     * @return The activation value of the neuron.
     */
    double getActivation() const; 

    /**
     * @brief Sets the gradient for backpropagation.
     * @param gradient The gradient value to set.
     */
    void setGradient(double gradient); 

    /**
     * @brief Gets the gradient for backpropagation.
     * @return The gradient value of the neuron.
     */
    double getGradient() const;

    /**
     * @brief Sets the weights connecting to the previous layer.
     * @param weights A vector of weights to set.
     */
    void setWeights(const std::vector<double>& weights); 

    /**
     * @brief Gets the weights connecting to the previous layer.
     * @return A vector of weights of the neuron.
     */
    std::vector<double> getWeights() const; 

    /**
    * @brief Hyperbolic tangent activation function.
    * @param x The input value.
    * @return double The activated value.
    */
    static double tanhActivation(double x);

    /**
    * @brief Derivative of the hyperbolic tangent activation function.
    * @param x The input value.
    * @return double The derivative value.
    */
    static double tanhDerivative(double x);
};

} // namespace kl

#endif // NEURON_HPP

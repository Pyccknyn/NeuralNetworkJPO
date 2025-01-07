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
    * @brief Constructs a default Neuron with zero initialization.
    * Initializes m_value, m_bias, m_activation, m_gradient to 0 and m_weights to an empty vector.
    */
    Neuron();

    void setValue(double value); ///< Sets the input value.
    double getValue() const; ///< Gets the input value.

    void setBias(double bias); ///< Sets the bias.
    double getBias() const; ///< Gets the bias.

    void setActivation(double activation); ///< Sets the activation value.
    double getActivation() const; ///< Gets the activation value.

    void setGradient(double gradient); ///< Sets the gradient.
    double getGradient() const; ///< Gets the gradient.

    void setWeights(const std::vector<double>& weights); ///< Sets the weights.
    std::vector<double> getWeights() const; ///< Gets the weights.


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

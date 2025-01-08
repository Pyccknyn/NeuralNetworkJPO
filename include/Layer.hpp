/**
 * @file Layer.hpp
 * @brief Defines the Layer base class and derived classes InputLayer, HiddenLayer, OutputLayer.
 */

#ifndef LAYER_HPP
#define LAYER_HPP

#include <vector>
#include <Eigen/Dense>
#include "Neuron.hpp"

namespace kl {

/**
 * @class Layer
 * @brief Base class representing a generic layer in the neural network.
 */
class Layer {
public:
    /**
     * @brief Virtual destructor for the base Layer class.
     */
    virtual ~Layer() = default;

    /**
     * @brief Pure virtual function for forward pass computation.
     */
    virtual void forward() = 0;

    /**
     * @brief Pure virtual function for backward pass computation.
     */
    virtual void backward() = 0;

    /**
     * @brief Gets a constant reference to the neurons in the layer.
     * @return A constant reference to the vector of neurons.
     */
    const std::vector<Neuron>& getNeurons() const;

    /**
     * @brief Gets a mutable reference to the neurons in the layer.
     * @return A reference to the vector of neurons.
     */
    std::vector<Neuron>& getNeurons();

protected:
    std::vector<Neuron> m_neurons; ///< Neurons in the layer.
    Layer* m_previousLayer = nullptr; ///< Pointer to the previous layer in the network.
    Layer* m_nextLayer = nullptr; ///< Pointer to the next layer in the network.
};

/**
 * @class InputLayer
 * @brief Represents the input layer of the neural network.
 */
class InputLayer : public Layer {
public:
    /**
     * @brief Constructs an InputLayer with the specified number of neurons.
     * @param numNeurons Number of neurons in the input layer.
     */
    explicit InputLayer(int numNeurons);

    /**
     * @brief Performs a forward pass with input values.
     * @param input A vector of input values for the layer.
     */
    void forwardParam(const Eigen::VectorXd& input);

    /**
     * @brief Performs a forward pass (overridden from the base class).
     */
    void forward() override;

    /**
     * @brief Performs a backward pass (overridden from the base class).
     */
    void backward() override;
};

/**
 * @class HiddenLayer
 * @brief Represents a hidden layer in the neural network.
 */
class HiddenLayer : public Layer {
public:
    /**
     * @brief Constructs a HiddenLayer with the specified number of neurons.
     * @param numNeurons Number of neurons in the hidden layer.
     */
    explicit HiddenLayer(int numNeurons);

    /**
     * @brief Performs a forward pass (overridden from the base class).
     */
    void forward() override;

    /**
     * @brief Performs a backward pass (overridden from the base class).
     */
    void backward() override;
};

/**
 * @class OutputLayer
 * @brief Represents the output layer of the neural network.
 */
class OutputLayer : public Layer {
public:
    /**
     * @brief Constructs an OutputLayer with the specified number of neurons.
     * @param numNeurons Number of neurons in the output layer.
     */
    explicit OutputLayer(int numNeurons);

    /**
     * @brief Performs a forward pass (overridden from the base class).
     */
    void forward() override;

    /**
     * @brief Performs a backward pass with target values.
     * @param target A vector of target values for backpropagation.
     */
    void backwardParam(const Eigen::VectorXd& target);

    /**
     * @brief Performs a backward pass (overridden from the base class).
     */
    void backward() override;
};

} // namespace kl

#endif // LAYER_HPP

# NeuralNetworkJPO

A student project developed for the Object-Oriented Programming (OOP) course.

The Doxygen documentation for this project is available [here](#).

## Introduction

Welcome to the NeuralNetworkJPO project documentation. This project is a simple implementation of a neural network designed for educational purposes. It includes key features such as:
- Defining and managing layers (input, hidden, and output).
- Forward propagation for predictions.
- Backpropagation for training the network.
- Customizable learning rate and topology.

## Features
- **Object-oriented design**: Layers and neurons are modular, following OOP principles.
- **Customizable**: Define your own network topology by specifying the number of neurons per layer.
- **Efficient training**: Includes gradient-based backpropagation for learning.
- **Flexible input/output**: Utilizes Eigen for vectorized input and output.

## Building

### Prerequisites

This application requires the following dependencies:
- Eigen library for linear algebra operations.
- C++17 compiler.

### Linux

#### Clone the Repository

    git clone https://github.com/example/NeuralNetworkJPO.git
    cd NeuralNetworkJPO

#### Build the Project Using CMake

A simple way to build this project is with CMake:

    mkdir build
    cd build
    cmake ..
    make
    ./NeuralNetworkJPO

#### Additional Notes

Ensure you have CMake installed. If not, install it using:

    sudo apt-get update
    sudo apt-get install cmake

Ensure you have Eigen3 installed. If not, install it using:

    sudo apt-get update
    sudo apt-get install libeigen3-dev

### Windows

1. Install a C++17-compatible compiler (e.g., Visual Studio).
2. Install CMake from [cmake.org](https://cmake.org/).
3. Download the desired release of Eigen from [https://eigen.tuxfamily.org/index.php?title=Main_Page](http://eigen.tuxfamily.org/).
4. Unzip in the location of your choice, preferrably at C:\ or C:\Program files for better discoverability by CMake find-modules (remember to extract the inner folder and rename it to Eigen3 or Eigen).
5. Follow the same steps as in the Linux section using a terminal like PowerShell or the CMake GUI.

## Usage

### Example Code

Here is an example of using the neural network to train and predict:

```cpp
#include "NeuralNetwork.hpp"
#include <Eigen/Dense>
#include <iostream>

int main() {
    // Define topology: 2 input neurons, 2 hidden neurons, and 1 output neuron
    std::vector<uint> topology = {2, 2, 1};
    double learningRate = 0.01;

    kl::NeuralNetwork nn(topology, learningRate);

    // Input and target data
    Eigen::VectorXd input(2);
    input << 1.0, 0.0;
    Eigen::VectorXd target(1);
    target << 1.0;

    // Number of epochs to train the network
    int epochs = 1000;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward propagation
        nn.forwardPropagation(input);

        // Backpropagation
        nn.backPropagation(target);

        // Update weights and biases
        nn.updateWeightsAndBiases();
    }
    // Predict output
    Eigen::VectorXd output = nn.predict(input);
    std::cout << "Predicted output: " << output[0] << std::endl;

    return 0;
}
```

## Example Execution

Train the network with custom data and observe the results:

```bash
$ ./NeuralNetworkJPO
Predicted output: 0.98765
```

## Author

This project was developed by Kacper Łucki as part of the Object-Oriented Programming course.

## License

This project is licensed under the MIT License.

/**
 * @file utils.cpp
 * @brief Utility functions for data preprocessing in the neural network.
 * 
 * This file contains helper functions for normalizing data to prepare it for neural network training.
 */

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "utils.hpp"


/**
 * @brief Normalizes a single input vector based on the provided reference matrix.
 * 
 * This function takes an input vector and normalizes each element based on the column-wise minimum
 * and maximum values in a reference matrix. The normalization maps values to the range [0, 1].
 * 
 * @param input The input vector to normalize.
 * @param reference The reference matrix to determine the minimum and maximum values for normalization.
 * @return Eigen::VectorXd The normalized input vector.
 * 
 * @note If the maximum and minimum values for a column are the same, the corresponding normalized value is set to 0.
 */
Eigen::VectorXd normalizeInput(const Eigen::VectorXd& input, const Eigen::MatrixXd& reference) {
    Eigen::VectorXd normalizedInput = input;
    for (int col = 0; col < input.size(); ++col) {
        double minVal = reference.col(col).minCoeff();
        double maxVal = reference.col(col).maxCoeff();
        if (maxVal != minVal) {
            normalizedInput[col] = (input[col] - minVal) / (maxVal - minVal);
        } else {
            normalizedInput[col] = 0.0; // Handle constant column
        }
    }
    return normalizedInput;
}

/**
 * @brief Normalizes all elements of a matrix column-wise to the range [0, 1].
 * 
 * This function normalizes each column of the given matrix by subtracting the column's minimum value
 * and dividing by the range (max - min). The normalization maps all values in the matrix to the range [0, 1].
 * 
 * @param matrix The matrix to normalize.
 * @return Eigen::MatrixXd The normalized matrix.
 * 
 * @note If the maximum and minimum values for a column are the same, the corresponding column is set to 0.
 */
Eigen::MatrixXd normalizeMatrix(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd normalizedMatrix = matrix;
    for (int col = 0; col < matrix.cols(); ++col) {
        double minVal = matrix.col(col).minCoeff();
        double maxVal = matrix.col(col).maxCoeff();
        if (maxVal != minVal) {
            normalizedMatrix.col(col) = (matrix.col(col).array() - minVal) / (maxVal - minVal);
        } else {
            normalizedMatrix.col(col).setZero(); // Handle constant column
        }
    }
    return normalizedMatrix;
}

/**
 * @brief Reads data from a CSV file and uploads it into an Eigen::MatrixXd.
 * @param filename The name of the CSV file.
 * @return Eigen::MatrixXd containing the CSV data.
 * @throws std::runtime_error If the file cannot be opened.
 */
Eigen::MatrixXd readCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    // Read each line from the CSV file
    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> row;

        // Read each cell in the line
        while (std::getline(lineStream, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                continue; // Skip invalid row
            }
        }
        data.push_back(row);
    }

    file.close();

    // Convert the data vector to an Eigen::MatrixXd
    if (data.empty()) {
        throw std::runtime_error("CSV file is empty.");
    }

    size_t numRows = data.size();
    size_t numCols = data[0].size();
    Eigen::MatrixXd matrix(numRows, numCols);

    for (size_t i = 0; i < numRows; ++i) {
        if (data[i].size() != numCols) {
            throw std::runtime_error("Inconsistent number of columns in CSV file.");
        }
        for (size_t j = 0; j < numCols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }

    return matrix;
}

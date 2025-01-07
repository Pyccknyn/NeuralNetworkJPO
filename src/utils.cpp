/**
 * @file utils.cpp
 * @brief Utility functions for data preprocessing in the neural network.
 * 
 * This file contains helper functions for normalizing data to prepare it for neural network training.
 */

#include <Eigen/Dense>
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


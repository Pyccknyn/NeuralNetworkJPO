#include <Eigen/Dense>
#include "utils.hpp"

// Function to normalize a single vector based on a reference matrix
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

// Function to normalize an entire matrix
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


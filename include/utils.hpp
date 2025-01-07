#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>

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
Eigen::VectorXd normalizeInput(const Eigen::VectorXd& input, const Eigen::MatrixXd& reference);

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
Eigen::MatrixXd normalizeMatrix(const Eigen::MatrixXd& matrix);



#endif // UTILS_HPP
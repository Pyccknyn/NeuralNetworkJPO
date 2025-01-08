/**
 * @file utils.hpp
 * @brief Utility functions for neural network operations.
 * 
 * This file contains utility functions for normalizing input vectors and matrices,
 * calculating mean squared error, and splitting datasets into training and testing sets.
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <vector> 

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

/**
 * @brief Reads data from a CSV file and uploads it into an Eigen::MatrixXd.
 * @param filename The name of the CSV file.
 * @return Eigen::MatrixXd containing the CSV data.
 * @throws std::runtime_error If the file cannot be opened.
 */
Eigen::MatrixXd readCSV(const std::string& filename);

#endif // UTILS_HPP
#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>

Eigen::VectorXd normalizeInput(const Eigen::VectorXd& input, const Eigen::MatrixXd& reference);

Eigen::MatrixXd normalizeMatrix(const Eigen::MatrixXd& matrix);

#endif // UTILS_HPP
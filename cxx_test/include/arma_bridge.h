#pragma once
#include <armadillo>

namespace org {
namespace armadillo {

struct Mat {
    arma::Mat<double> mat;
};

std::unique_ptr<Mat> arma_matmul(const double* a, const double* b, int m, int k, int n);
void transpose(Mat& a);
double* mat_data(const std::unique_ptr<Mat>& a);


} // namespace armadillo
} // namespace org
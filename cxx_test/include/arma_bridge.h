#pragma once
#include "rust/cxx.h"
#include <armadillo>

namespace org {
    namespace armadillo {

        struct Mat {
            arma::Mat<double> mat;
        };

        struct RustMat;

        std::unique_ptr<Mat> arma_matmul(const double* a, const double* b, int m, int k, int n);
        void transpose(Mat& a);
        double* mat_data(const std::unique_ptr<Mat>& a);
        rust::Box<RustMat> transpose_and_raise(Mat& a);


    } // namespace armadillo
} // namespace org
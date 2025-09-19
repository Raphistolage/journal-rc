#pragma once
#include <cstddef>

extern "C" {

// Multiplies two matrices (A: m x k, B: k x n), result in C (m x n)
// All matrices are row-major, contiguous arrays
void arma_matmul(const double* a, const double* b, double* c, int m, int k, int n);

// Adds k to each element of the matrix (in place)
void raise_mat(double* a, int len, double k);

}
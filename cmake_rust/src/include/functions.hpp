#pragma once
#include "cxx.h"
#include <Kokkos_Core.hpp>
#include "ffi.rs.h"

int perf_y_ax(rust::Vec<rust::String> argv);

void checkSizes( int &N, int &M, int &S, int &nrepeat );
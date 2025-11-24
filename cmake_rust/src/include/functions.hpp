#pragma once
#include "cxx.h"
#include <Kokkos_Core.hpp>

namespace cpp_functions
{
    void kokkos_initialize();
    void kokkos_finalize();

    void parallel_hello_world();
} // namespace cpp_functions

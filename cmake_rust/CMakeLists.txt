cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

find_package(Kokkos REQUIRED PATHS /home/clissonr/Documents/StageCEA/cmake_rust/kokkos-install NO_DEFAULT_PATH)

# find_package(OpenMP)

add_library(cxxKokkoslib SHARED src/cpp/functions.cpp target/cxxbridge/cmake_rust/src/ffi.rs.cc)

set_target_properties(cxxKokkoslib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cxxKokkoslib PUBLIC Kokkos::kokkos)

target_include_directories(cxxKokkoslib PUBLIC 
    target/cxxbridge/rust
    src/include
    kokkos-install/include 
    kokkos-install/lib64
)
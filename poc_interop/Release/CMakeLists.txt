cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

find_package(Kokkos REQUIRED PATHS /home/clissonr/Documents/StageCEA/journal-rc/kokkos-install NO_DEFAULT_PATH) # On va chercher l'installation locale.




# RustView : 
add_library(rustView SHARED ../src/cpp/rust_view.cpp ../target/cxxbridge/poc_interop/src/rust_view/ffi.rs.cc)

target_link_libraries(rustView PUBLIC Kokkos::kokkos)

target_include_directories(rustView PUBLIC     
    ../target/cxxbridge/rust
    ../target/cxxbridge/poc_interop/src/rust_view/
    ../src/include
)

set_target_properties(rustView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Functions ffi : 
add_library(functionsFfi SHARED 
    ../src/cpp/functions.cpp 
    ../target/cxxbridge/poc_interop/${OUT_DIR}/create_view_ffi.rs.cc
    ../target/cxxbridge/poc_interop/${OUT_DIR}/dot_ffi.rs.cc
    ../target/cxxbridge/poc_interop/${OUT_DIR}/get_ffi.rs.cc
)

target_link_libraries(functionsFfi PUBLIC Kokkos::kokkos rustView) # les dependances de rustView vont transiter à functionsFfi, ca devrait être bon.

target_include_directories(functionsFfi PUBLIC ../target/cxxbridge/poc_interop/${OUT_DIR})

set_target_properties(functionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedArrayView : 
add_library(sharedArrayView SHARED ../src/cpp/shared_array.cpp)

target_link_libraries(sharedArrayView PUBLIC Kokkos::kokkos)

target_include_directories(sharedArrayView PUBLIC     
    ../src/include
)

set_target_properties(sharedArrayView PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS rustView DESTINATION .)
install(TARGETS functionsFfi DESTINATION .)
install(TARGETS sharedArrayView DESTINATION .)
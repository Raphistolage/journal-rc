cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

find_package(Kokkos REQUIRED PATHS ../../kokkos-install NO_DEFAULT_PATH) # On va chercher l'installation locale.


add_library(kokkosHandles SHARED ../src/cpp/kokkos_handles.cpp)

target_link_libraries(kokkosHandles PUBLIC Kokkos::kokkos)

set_target_properties(kokkosHandles PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedArrayView
add_library(sharedArrayView SHARED ../src/cpp/shared_array.cpp)

target_link_libraries(sharedArrayView PUBLIC kokkosHandles)

target_include_directories(sharedArrayView PUBLIC     
    ../src/include
)

set_target_properties(sharedArrayView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView
add_library(rustView SHARED ../src/cpp/rust_view.cpp ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/rust_view/ffi.rs.cc)

target_link_libraries(rustView PUBLIC kokkosHandles)

target_include_directories(rustView PUBLIC     
    ${OUT_DIR}/cxxbridge/include/rust
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/rust_view/
    ../src/include
)

set_target_properties(rustView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Functions ffi
add_library(functionsFfi SHARED 
    ../src/cpp/functions.cpp 
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${OUT_DIR}/test_ffi_ffi.rs.cc
)

target_link_libraries(functionsFfi PUBLIC rustView)

target_include_directories(functionsFfi PUBLIC ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/)

set_target_properties(functionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS kokkosHandles DESTINATION .)
install(TARGETS sharedArrayView DESTINATION .)
install(TARGETS rustView DESTINATION .)
install(TARGETS functionsFfi DESTINATION .)
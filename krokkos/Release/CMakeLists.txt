cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  if(EXISTS ${Kokkos_COMMON_SOURCE_DIR})
    add_subdirectory(${Kokkos_COMMON_SOURCE_DIR} Kokkos)
  else()
    include(FetchContent)
    FetchContent_Declare(
      Kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG        5.0.0
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()


# rustViewFfiTypes

add_library(rustViewFfiTypes SHARED ${OUT_DIR}/cxxbridge/sources/krokkos/src/rust_view/shared_ffi_types.rs.cc)

target_include_directories(rustViewFfiTypes PUBLIC 
${OUT_DIR}/cxxbridge/include/krokkos/src/rust_view/ 
${OUT_DIR}/cxxbridge/include/rust
../src/include
)

set_target_properties(rustViewFfiTypes PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView
add_library(rustView SHARED ../src/cpp/rust_view.cpp ${OUT_DIR}/cxxbridge/sources/krokkos/src/rust_view/ffi.rs.cc)

target_link_libraries(rustView PUBLIC Kokkos::kokkos rustViewFfiTypes)

target_include_directories(rustView PUBLIC     
  ${OUT_DIR}/cxxbridge/krokkos/src/rust_view/ 
  ${OUT_DIR}/cxxbridge/rust/
    ../src/include
)

set_target_properties(rustView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView_Functions ffi
add_library(rustViewFunctionsFfi SHARED 
    ${OUT_DIR}/cxxbridge/sources/krokkos/${OUT_DIR}/../../../../krokkosbridge/rust_view_functions_ffi.rs.cc
    ../src/cpp/functions.cpp
)

target_link_libraries(rustViewFunctionsFfi PUBLIC Kokkos::kokkos rustView )

target_include_directories(rustViewFunctionsFfi PUBLIC ${OUT_DIR}/cxxbridge/include/krokkos/${OUT_DIR}/../../../../krokkosbridge/)

set_target_properties(rustViewFunctionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)


install(TARGETS rustView DESTINATION .)
install(TARGETS rustViewFfiTypes DESTINATION .)
install(TARGETS rustViewFunctionsFfi DESTINATION .)

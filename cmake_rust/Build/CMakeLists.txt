cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

# find_package(Kokkos REQUIRED PATHS /home/clissonr/Documents/StageCEA/journal-rc/kokkos-install NO_DEFAULT_PATH) # On va chercher l'installation locale.

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
      GIT_TAG        4.7.01
      SOURCE_DIR ${Kokkos_COMMON_SOURCE_DIR}
    )
    FetchContent_MakeAvailable(Kokkos)
  endif()
endif()

# find_package(gomp)

add_library(cxxKokkoslib SHARED ../src/cpp/functions.cpp ../target/cxxbridge/cmake_rust/src/ffi.rs.cc)

set_target_properties(cxxKokkoslib PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(cxxKokkoslib PUBLIC Kokkos::kokkos)

target_include_directories(cxxKokkoslib PUBLIC 
    ../target/cxxbridge/rust
    ../src/include
    ../target/cxxbridge/cmake_rust/src/
)

install(TARGETS cxxKokkoslib DESTINATION .)
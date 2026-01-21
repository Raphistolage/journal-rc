cmake_minimum_required(VERSION 3.16)
project(krokkos CXX)

find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
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

# Krokkos

add_library(krokkos SHARED ${OUT_DIR}/cxxbridge/${PKG_NAME}/${OUT_DIR}/krokkosbridge/krokkos_bridge.rs.cc)

target_link_libraries(krokkos PUBLIC Kokkos::kokkos)

target_include_directories(krokkos PUBLIC 
${OUT_DIR}/cxxbridge/rust/
${OUT_DIR}/cxxbridge/${PKG_NAME}/${OUT_DIR}/krokkosbridge/
${OUT_DIR}/krokkosbridge/
)

install(TARGETS krokkos DESTINATION .)
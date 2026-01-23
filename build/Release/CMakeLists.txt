cmake_minimum_required(VERSION 3.16)
project(krokkos CXX)

find_package(Kokkos CONFIG)
if(Kokkos_FOUND)
  message(STATUS "Found Kokkos: ${Kokkos_DIR} (version \"${Kokkos_VERSION}\")")
else()
  include(FetchContent)
  FetchContent_Declare(
    Kokkos
    URL https://github.com/kokkos/kokkos/archive/refs/tags/5.0.0.zip
  )
  FetchContent_MakeAvailable(Kokkos)
endif()

# Krokkos

add_library(krokkos SHARED ${TARGET_DIR}/cxxbridge/${PKG_NAME}/${TARGET_DIR}/krokkosbridge/krokkos_bridge.rs.cc)

target_link_libraries(krokkos PUBLIC Kokkos::kokkos)

target_include_directories(krokkos PUBLIC 
${TARGET_DIR}/cxxbridge/rust/
${TARGET_DIR}/cxxbridge/${PKG_NAME}/${TARGET_DIR}/krokkosbridge/
${TARGET_DIR}/krokkosbridge/
)

install(TARGETS krokkos DESTINATION .)
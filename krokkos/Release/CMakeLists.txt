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


# KokkosHandles (sert a linker kokkos qu'une seule fois, un seul contexte)
add_library(kokkosHandles SHARED ../src/cpp/kokkos_handles.cpp)

target_link_libraries(kokkosHandles PUBLIC Kokkos::kokkos)

set_target_properties(kokkosHandles PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedArray
add_library(sharedArray SHARED ../src/cpp/shared_array.cpp ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/shared_array/ffi.rs.cc)

target_link_libraries(sharedArray PUBLIC kokkosHandles)

target_include_directories(sharedArray PUBLIC 
    ${OUT_DIR}/cxxbridge/include/rust
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/shared_array/    
    ../src/include
)

set_target_properties(sharedArray PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
    ../src/cpp/functions_shared_array.cpp
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${OUT_DIR}/rust_view_ffi_ffi.rs.cc
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${OUT_DIR}/shared_array_ffi_ffi.rs.cc
)

target_link_libraries(functionsFfi PUBLIC rustView sharedArray)

target_include_directories(functionsFfi PUBLIC ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/)

set_target_properties(functionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS kokkosHandles DESTINATION .)
install(TARGETS sharedArray DESTINATION .)
install(TARGETS rustView DESTINATION .)
install(TARGETS functionsFfi DESTINATION .)
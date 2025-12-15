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

# SharedFfiTypes

add_library(sharedFfiTypes SHARED ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/shared_array/shared_ffi_types.rs.cc)
target_include_directories(sharedFfiTypes PUBLIC ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/shared_array/)

set_target_properties(sharedFfiTypes PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedRustViewFfiTypes

add_library(sharedRustViewFfiTypes SHARED ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/rust_view/shared_ffi_types.rs.cc)

target_include_directories(sharedRustViewFfiTypes PUBLIC 
${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/rust_view/ 
${OUT_DIR}/cxxbridge/include/rust
../src/include
)

set_target_properties(sharedRustViewFfiTypes PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView
add_library(rustView SHARED ../src/cpp/rust_view.cpp ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/rust_view/ffi.rs.cc)

target_link_libraries(rustView PUBLIC kokkosHandles sharedRustViewFfiTypes)

target_include_directories(rustView PUBLIC     
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/rust_view/
    ${OUT_DIR}/cxxbridge/include/rust
    ../src/include
)

set_target_properties(rustView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedArray_Functions ffi
add_library(sharedArrayFunctionsFfi SHARED 
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${OUT_DIR}/shared_array_functions_ffi.rs.cc
    ../src/cpp/functions_shared_array.cpp
)

target_link_libraries(sharedArrayFunctionsFfi PUBLIC kokkosHandles sharedFfiTypes)

target_include_directories(sharedArrayFunctionsFfi PUBLIC 
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/
    ../src/include/
)

set_target_properties(sharedArrayFunctionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)

# SharedArray
add_library(sharedArray SHARED ../src/cpp/shared_array.cpp ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/src/shared_array/ffi.rs.cc)

target_link_libraries(sharedArray PUBLIC sharedArrayFunctionsFfi)

target_include_directories(sharedArray PUBLIC 
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/src/shared_array/   
    ${OUT_DIR}/cxxbridge/include/rust

    ../src/include/
)

set_target_properties(sharedArray PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView_Functions ffi
add_library(rustViewFunctionsFfi SHARED 
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${OUT_DIR}/rust_view_functions_ffi.rs.cc
    ../src/cpp/functions.cpp
)

target_link_libraries(rustViewFunctionsFfi PUBLIC rustView )

target_include_directories(rustViewFunctionsFfi PUBLIC ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/)

set_target_properties(rustViewFunctionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS kokkosHandles DESTINATION .)
install(TARGETS sharedArray DESTINATION .)
install(TARGETS sharedArrayFunctionsFfi DESTINATION .)
install(TARGETS rustView DESTINATION .)
install(TARGETS sharedRustViewFfiTypes DESTINATION .)
install(TARGETS rustViewFunctionsFfi DESTINATION .)

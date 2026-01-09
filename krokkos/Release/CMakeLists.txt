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

add_library(rustViewFfiTypes SHARED ${OUT_DIR}/../../../../cxxbridge/krokkos/src/rust_view/shared_ffi_types.rs.cc)

target_include_directories(rustViewFfiTypes PUBLIC 
${OUT_DIR}/../../../../cxxbridge/krokkos/src/rust_view/ 
${OUT_DIR}/../../../../cxxbridge/rust/
../src/include
)

set_target_properties(rustViewFfiTypes PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView
add_library(rustView SHARED ../src/cpp/rust_view.cpp ${OUT_DIR}/../../../../cxxbridge/krokkos/src/rust_view/ffi.rs.cc)

target_link_libraries(rustView PUBLIC Kokkos::kokkos rustViewFfiTypes)

target_include_directories(rustView PUBLIC     
${OUT_DIR}/../../../../cxxbridge/krokkos/src/rust_view/ 
${OUT_DIR}/../../../../cxxbridge/rust/
    ../src/include
)

set_target_properties(rustView PROPERTIES POSITION_INDEPENDENT_CODE ON)

# RustView_Functions ffi
add_library(rustViewFunctionsFfi SHARED 
    ${OUT_DIR}/../../../../cxxbridge/krokkos/${OUT_DIR}/../../../../krokkosbridge/rust_view_functions_ffi.rs.cc
    ../src/cpp/functions.cpp
)

target_link_libraries(rustViewFunctionsFfi PUBLIC Kokkos::kokkos rustView )

target_include_directories(rustViewFunctionsFfi PUBLIC ${OUT_DIR}/../../../../cxxbridge/krokkos/${OUT_DIR}/../../../../krokkosbridge/)

set_target_properties(rustViewFunctionsFfi PROPERTIES POSITION_INDEPENDENT_CODE ON)


# User's cpp file 
if(DEFINED USER_CPP_FILE_PATH)
  add_library(userCppFunctions SHARED 
    ${USER_MANIFEST_DIR}/${USER_CPP_FILE_PATH}
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${USER_RUST_FFI_FILE_PATH}.cc  
  )

  target_link_libraries(userCppFunctions PUBLIC rustView rustViewFfiTypes)

  target_include_directories(userCppFunctions PUBLIC 
    ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/
    ${USER_MANIFEST_DIR}/${CPP_FOLDER}/
    ${OUT_DIR}/../../../../cxxbridge/rust/
  )

  set_target_properties(userCppFunctions PROPERTIES POSITION_INDEPENDENT_CODE ON)

  install(TARGETS userCppFunctions DESTINATION .)
endif()

install(TARGETS rustView DESTINATION .)
install(TARGETS rustViewFfiTypes DESTINATION .)
install(TARGETS rustViewFunctionsFfi DESTINATION .)
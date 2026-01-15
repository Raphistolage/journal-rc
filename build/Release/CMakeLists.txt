cmake_minimum_required(VERSION 3.16)
project(Krokkos CXX)

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

# Krokkos

add_library(Krokkos SHARED ${OUT_DIR}/../../../../cxxbridge/${PKG_NAME}/${OUT_DIR}/../../../../krokkosbridge/krokkos_bridge.rs.cc)

target_link_libraries(Krokkos PUBLIC Kokkos::kokkos)

target_include_directories(Krokkos PUBLIC 
  ${OUT_DIR}/../../../../cxxbridge/rust/
  ${OUT_DIR}/../../../../cxxbridge/${PKG_NAME}/${OUT_DIR}/../../../../krokkosbridge/
  /home/clissonr/Documents/StageCEA/Krokkos/test_macros/target/cxxbridge/test_macros/home/clissonr/Documents/StageCEA/Krokkos/test_macros/target/krokkosbridge/
  ${OUT_DIR}/../../../../krokkosbridge/
)

if(DEFINED USER_CPP_FILE_PATH)
add_library(userCppFunctions SHARED 
${USER_MANIFEST_DIR}/${USER_CPP_FILE_PATH}
${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${USER_RUST_FFI_FILE_PATH}.cc  
)

target_link_libraries(userCppFunctions PUBLIC Krokkos)

target_include_directories(userCppFunctions PUBLIC 
${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/
${USER_MANIFEST_DIR}/${CPP_FOLDER}/
${OUT_DIR}/../../../../cxxbridge/rust/
${OUT_DIR}/../../../../krokkosbridge/
)

set_target_properties(userCppFunctions PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS userCppFunctions DESTINATION .)
endif()

install(TARGETS Krokkos DESTINATION .)
cmake_minimum_required(VERSION 3.16)
project(cxxKokkoslib CXX)

find_package(kokkosHandles CONFIG)
if(kokkosHandles_FOUND)
  message(STATUS "Found kokkosHandles")
endif()

find_package(rustView CONFIG)
if(rustView_FOUND)
  message(STATUS "Found rustView")
endif()

find_package(sharedRustViewFfiTypes CONFIG)
if(sharedRustViewFfiTypes_FOUND)
  message(STATUS "Found sharedRustViewFfiTypes")
endif()

# Extra user's functions (if defined)
if(DEFINED USER_CPP_FILE_PATH)
  add_library(userCppFunctions SHARED 
    ../${USER_CPP_FILE_PATH}
    ${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${USER_RUST_FFI_FILE_PATH}.cc  
  )

  target_link_libraries(userCppFunctions PUBLIC kokkosHandles rustView sharedRustViewFfiTypes)

  target_include_directories(userCppFunctions PUBLIC ${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/)

  set_target_properties(userCppFunctions PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(DEFINED USER_CPP_FILE_PATH)
  install(TARGETS userCppFunctions DESTINATION .)
endif()
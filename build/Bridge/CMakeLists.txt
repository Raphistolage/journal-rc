cmake_minimum_required(VERSION 3.16)
project(userKrokkos CXX)

add_library(Krokkos SHARED IMPORTED)
set_property(TARGET Krokkos PROPERTY IMPORTED_LOCATION ${OUT_DIR}/libKrokkos.so)

add_library(userCppFunctions SHARED 
${USER_MANIFEST_DIR}/${USER_CPP_FILE_PATH}
${OUT_DIR}/cxxbridge/sources/${PKG_NAME}/${USER_RUST_FFI_FILE_PATH}.cc  
)

target_link_libraries(userCppFunctions PUBLIC Krokkos)

target_include_directories(userCppFunctions PUBLIC 
${OUT_DIR}/cxxbridge/include/${PKG_NAME}/${OUT_DIR}/
${USER_MANIFEST_DIR}/${CPP_FOLDER}/
${OUT_DIR}/../../../../cxxbridge/rust/
)

set_target_properties(userCppFunctions PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS userCppFunctions DESTINATION .)

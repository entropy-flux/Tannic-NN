cmake_minimum_required(VERSION 3.28)
project(tannic-nn LANGUAGES CXX)
 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/tannic/cmake")
include_directories("${CMAKE_SOURCE_DIR}/external/tannic/include")
include_directories("${CMAKE_SOURCE_DIR}/external/tannic/include/tannic")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ENABLED TRUE)
    message(STATUS "CUDA support enabled.")
    find_package(CUDAToolkit)
else()
    message(STATUS "CUDA not found. Building without CUDA support.")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CUDA_ENABLED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
 
find_library(TANNIC_LIB tannic PATHS "${CMAKE_SOURCE_DIR}/external/tannic/build" NO_DEFAULT_PATH)
if(NOT TANNIC_LIB)
    message(STATUS "tannic library not found, building it...")
    add_subdirectory(external/tannic)
    set(TANNIC_LIB tannic)
endif()

include_directories(include/tannic-nn)
 
file(GLOB_RECURSE SRC_FILES
    src/*.cpp
    src/cpu/*.cpp
    src/cuda/*.cu
)
 
# Robust OpenBLAS detection
find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS /usr/include /usr/local/include /opt/openblas/include
    DOC "OpenBLAS include directory"
)

find_library(OPENBLAS_LIBRARY
    NAMES openblas openblas64
    PATHS /usr/lib /usr/lib64 /usr/local/lib /opt/openblas/lib
    DOC "OpenBLAS library"
)

if(OPENBLAS_INCLUDE_DIR AND OPENBLAS_LIBRARY AND EXISTS "${OPENBLAS_INCLUDE_DIR}/cblas.h")
    set(OpenBLAS_FOUND TRUE)
    set(OpenBLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIR})
    set(OpenBLAS_LIBRARIES ${OPENBLAS_LIBRARY})
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS}")
else()
    message(STATUS "OpenBLAS not found, continuing without it.")
    set(OpenBLAS_FOUND FALSE)
endif()
 
add_library(tannic-nn STATIC ${SRC_FILES})
target_include_directories(tannic-nn PUBLIC 
    include
    "${CMAKE_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/src/cpu"
    "${CMAKE_SOURCE_DIR}/src/cuda"
    "${CMAKE_SOURCE_DIR}/external/tannic/include"
    "${CMAKE_SOURCE_DIR}/external/tannic/include/tannic"
)
  
target_link_libraries(tannic-nn PUBLIC ${TANNIC_LIB})
 
if(CUDA_ENABLED)
    set_target_properties(tannic-nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tannic-nn PUBLIC CUDA::cudart)
    target_compile_definitions(tannic-nn PUBLIC CUDA)
endif()
 
if(OpenBLAS_FOUND)
    target_include_directories(tannic-nn PUBLIC ${OpenBLAS_INCLUDE_DIRS})
    target_link_libraries(tannic-nn PUBLIC ${OpenBLAS_LIBRARIES})
    target_compile_definitions(tannic-nn PUBLIC BLAS)
endif()
 
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GoogleTest not found, fetching...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)
else()
    message(STATUS "Using system-installed GTest")
endif()

file(GLOB_RECURSE TEST_FILES tests/*.cpp)

if(TEST_FILES)
    enable_testing()

    foreach(filename ${TEST_FILES})
        file(RELATIVE_PATH relpath ${CMAKE_SOURCE_DIR}/tests ${filename})
        string(REPLACE "/" "-" testname ${relpath})
        string(REPLACE ".cpp" "" testname ${testname})

        add_executable(${testname} ${filename})
        target_include_directories(${testname} PRIVATE include)
        target_link_libraries(${testname} PRIVATE tannic-nn GTest::gtest_main)

        if(CUDA_ENABLED)
            set_target_properties(${testname} PROPERTIES
                CUDA_SEPARABLE_COMPILATION ON
                CUDA_STANDARD 20
            )
            target_link_libraries(${testname} PRIVATE CUDA::cudart)
        endif()

        add_test(NAME ${testname} COMMAND ${testname})
    endforeach()
endif() 
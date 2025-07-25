cmake_minimum_required(VERSION 3.28)
project(tannic-nn LANGUAGES CXX)
 
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/external/tannic/cmake")
include_directories("${CMAKE_SOURCE_DIR}/external/tannic/include")

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
 
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS: ${OpenBLAS_INCLUDE_DIRS}")
    add_definitions(-DOPENBLAS)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
endif()
 
add_library(tannic-nn STATIC ${SRC_FILES})
target_include_directories(tannic-nn PUBLIC 
    include
    "${CMAKE_SOURCE_DIR}/external/tannic/include"
)
  
target_link_libraries(tannic-nn PUBLIC ${TANNIC_LIB})
 
if(CUDA_ENABLED)
    set_target_properties(tannic-nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(tannic-nn PUBLIC CUDA::cudart)
    target_compile_definitions(tannic-nn PUBLIC CUDA)
endif()
 
if(OpenBLAS_FOUND)
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
    add_executable(tannic-nn-tests ${TEST_FILES})
    target_include_directories(tannic-nn-tests PRIVATE 
        include
        "${CMAKE_SOURCE_DIR}/external/tannic/include"
    )
    target_link_libraries(tannic-nn-tests PRIVATE tannic-nn)

    if(CUDA_ENABLED)
        set_target_properties(tannic-nn-tests PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 20
        )
        target_link_libraries(tannic-nn-tests PRIVATE CUDA::cudart)
    endif()

    if(GTest_FOUND)
        target_link_libraries(tannic-nn-tests PRIVATE GTest::gtest_main)
    else()
        target_link_libraries(tannic-nn-tests PRIVATE gtest_main)
    endif()

    include(GoogleTest)
    gtest_discover_tests(tannic-nn-tests)
endif()
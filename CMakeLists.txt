cmake_minimum_required(VERSION 3.28)

project(tannic-nn LANGUAGES CXX)
 
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TANNIC_ENABLE_CUDA "Enable CUDA support" ON)
option(TANNIC_ENABLE_TESTS "Build tests" ON)
option(TANNIC_BUILD_MAIN "Build main.cpp executable" ON)
 
set(TANNIC_ROOT "${CMAKE_SOURCE_DIR}/external/tannic")
 
if(NOT TARGET tannic)
    add_subdirectory(${TANNIC_ROOT})
endif()
 
include(cmake/CMakeLists.txt)
 
if(TANNIC_BUILD_MAIN)
    add_executable(tannic-nn-main main.cpp)
    target_link_libraries(tannic-nn-main PRIVATE tannic-nn)
    target_include_directories(tannic-nn-main PRIVATE include "${TANNIC_ROOT}/include")
endif()
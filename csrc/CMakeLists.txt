cmake_minimum_required(VERSION 3.25)
set(USE_CUDNN 1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(stratego LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(COMPILATION_FLAGS "-DNDEBUG -Ofast -fopenmp -Wall -Wextra -Wfatal-errors -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS} ${COMPILATION_FLAGS}")

find_package(pybind11 REQUIRED)

add_library(traverser traverser.cpp)
add_executable(dh3 dh3.cpp)
target_link_libraries(dh3 traverser)

pybind11_add_module(pydh3 bindings.cpp)
target_link_libraries(pydh3 PRIVATE traverser)
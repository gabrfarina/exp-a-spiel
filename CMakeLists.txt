cmake_minimum_required(VERSION 3.29)
cmake_policy(SET CMP0167 NEW)
set(Boost_DEBUG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS headers)
message(STATUS "Found boost: ${Boost_FOUND} ${Boost_VERSION}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")

set(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wextra -Wfatal-errors -march=native -fPIC")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_DEBUG -DDEBUG -O2 -fno-omit-frame-pointer -g -fvisibility=hidden")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -Ofast -g -fvisibility=hidden")
endif()


add_subdirectory(csrc)
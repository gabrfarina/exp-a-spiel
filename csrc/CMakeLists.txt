add_library(traverser traverser.cpp averager.cpp cfr.cpp)
target_include_directories(traverser PUBLIC ${Boost_INCLUDE_DIRS})

add_executable(dh3 dh3.cpp)
target_link_libraries(dh3 traverser)

python_add_library(pydh3 MODULE bindings.cpp WITH_SOABI)
target_link_libraries(pydh3 PRIVATE traverser pybind11::headers)

# This is passing in the version as a define just as an example
target_compile_definitions(pydh3 PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS pydh3 DESTINATION ${SKBUILD_PROJECT_NAME})


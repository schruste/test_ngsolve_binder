cmake_minimum_required(VERSION 2.8.12)
project(min_dtn)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_subdirectory(pybind11)
pybind11_add_module(min_dtn src/full.cpp)

target_link_libraries(min_dtn PRIVATE ${CERES_LIBRARIES})

find_package (Python3 COMPONENTS Interpreter REQUIRED)
install(TARGETS min_dtn DESTINATION Python3_LIBRARY_DIRS)

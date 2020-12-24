# ED_libs
cmake_minimum_required(VERSION 3.3 FATAL_ERROR)

# Project
project(ED_Lib)

# Set output directories.
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# C++11
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

#
set(CMAKE_BUILD_TYPE "Debug")

# OpenCV
find_package(OpenCV REQUIRED)
message("-- OpenCV_VER: " ${OpenCV_VERSION})
message("-- OpenCV_DIR: " ${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


# Add lib
add_library(ED_Lib
    src/ED.cpp
    src/NFA.cpp
    src/EDPF.cpp
    src/EDCircles.cpp
    src/EDColor.cpp
    src/EDLines.cpp
)
target_link_libraries(ED_Lib
    ${OpenCV_LIBS}
)

# Exe
add_executable(ED_Lib_test test.cpp)
target_link_libraries(ED_Lib_test
    ED_Lib
)

# test runway
add_executable(test_runway test_runway.cpp)
target_link_libraries(test_runway
    ED_Lib
    stdc++fs    # <experimental/filesystem>
)



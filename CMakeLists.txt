cmake_minimum_required(VERSION 3.20)

project(cropgbc DESCRIPTION "Crop Game Boy Camera image." LANGUAGES CXX)

# To put executables next to the runtime libraries generated by conan.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# For fPIC / position independent code.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# C++ standard version.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adding subfolder property.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# External packages.
find_package(absl CONFIG REQUIRED)
find_package(opencv CONFIG REQUIRED)

# Sources.
add_subdirectory(sources)

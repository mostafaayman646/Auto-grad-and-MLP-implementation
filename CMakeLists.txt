cmake_minimum_required(VERSION 3.12)
project(AutoGrad)
cmake_policy(SET CMP0148 NEW)
set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)

# 1. Enforce EXACTLY Python 3.12
find_package(Python3 3.12 EXACT COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 REQUIRED)

file(GLOB SRC_FILES "Engine/*.cpp")
pybind11_add_module(CppModule ${SRC_FILES})
if(MINGW)
    target_link_options(CppModule PRIVATE "-static" "-static-libgcc" "-static-libstdc++")
endif()

set_target_properties(CppModule PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/Engine"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/Engine"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/Engine"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/Engine"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/Engine"
    OUTPUT_NAME "CppModule"
)

target_include_directories(CppModule PRIVATE utils)
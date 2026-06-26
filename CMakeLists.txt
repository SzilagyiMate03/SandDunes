cmake_minimum_required(VERSION 3.15)
project(dune_gpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Python + pybind11 ----
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# ---- OpenCL ----
find_package(OpenCL REQUIRED)

# ---- Shared library (Python modul) ---- 
pybind11_add_module(dune_gpu dune_gpu.cpp)

target_link_libraries(dune_gpu PRIVATE OpenCL::OpenCL)

target_compile_options(dune_gpu PRIVATE /O2 /W4)

add_custom_command(TARGET dune_gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:dune_gpu>
        ${CMAKE_SOURCE_DIR}
)

# ---- Telepítés (opcionális) ----
install(TARGETS dune_gpu
        LIBRARY DESTINATION ${Python3_SITEARCH})

# ---- Kernel fájl másolása a build könyvtárba ----
configure_file(
    ${CMAKE_SOURCE_DIR}/dune_gpu.cl
    ${CMAKE_BINARY_DIR}/dune_gpu.cl
    COPYONLY
)

message(STATUS "OpenCL found: ${OpenCL_LIBRARIES}")
message(STATUS "Python: ${Python3_EXECUTABLE}")

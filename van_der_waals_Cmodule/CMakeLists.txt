cmake_minimum_required(VERSION 3.20)
set(CMAKE_VERBOSE_MAKEFILE on)

project(van_der_waals_Cmodule)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(MATH_LIBS m gsl gslcblas)
set(PY_LIBS crypt pthread dl util rt python3.8)

find_package(PythonLibs REQUIRED)
include_directories(/home/ypolyach/anaconda3/include)
include_directories(/home/ypolyach/anaconda3/include/python3.8)
link_directories(/home/ypolyach/anaconda3/lib)

add_executable(van_der_waals_Cmodule main.cpp VdW.h VdW.cpp)
target_link_libraries(van_der_waals_Cmodule ${MATH_LIBS} ${PY_LIBS})

find_package(pybind11 REQUIRED)
pybind11_add_module(VdW.so VdW.h VdW.cpp VdW_py.cpp)
target_link_libraries(VdW.so PRIVATE ${MATH_LIBS} ${PY_LIBS})

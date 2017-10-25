
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-ignored-attributes -Wno-unused-parameter -Wno-unused-function")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ffast-math -fno-finite-math-only -march=native -Wall -Wno-ignored-attributes -Wextra -Wno-unused-parameter -Wno-unused-function")

# Require at least cmake 3.1 for find_package(OpenCL)
cmake_minimum_required (VERSION 3.1)
project (karpfen)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release"
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#find_package(OpenGL REQUIRED)
#find_package(GLEW REQUIRED)
#find_package(GLUT REQUIRED)

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS})

include_directories(${PROJECT_BINARY_DIR} ${OpenCL_INCLUDE_DIRS})

add_executable(karpfen_sp karpfen.cpp)
add_executable(karpfen_dp karpfen.cpp)

target_compile_definitions(karpfen_sp PRIVATE DOUBLE_PRECISION=0)
target_compile_definitions(karpfen_dp PRIVATE DOUBLE_PRECISION=1)

set(LIBS ${OpenCL_LIBRARIES} cfitsio)

target_link_libraries (karpfen_sp ${LIBS})
target_link_libraries (karpfen_dp ${LIBS})



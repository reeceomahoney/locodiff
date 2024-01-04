cmake_minimum_required(VERSION 3.28)
project(raisim_env)
set(CMAKE_CXX_STANDARD 17)

# ##############################################################################
# dependencies
# ##############################################################################

# python
set(Python_ROOT_DIR "/home/reece/miniconda3/envs/robodiff")
find_package(Python COMPONENTS Interpreter Development)

# raisim
list(APPEND CMAKE_PREFIX_PATH "$ENV{RAISIM_LIB}/raisim/linux")
find_package(raisim CONFIG REQUIRED)

# openmp
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# pybind
find_package(pybind11 REQUIRED)

# eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# ##############################################################################
# build
# ##############################################################################

pybind11_add_module(${PROJECT_NAME} raisim/raisim_gym.cpp raisim/Yaml.cpp)
target_compile_definitions(
  ${PROJECT_NAME}
  PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>" EIGEN_DONT_PARALLELIZE)
set_target_properties(
  ${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY
                             ${CMAKE_CURRENT_SOURCE_DIR}/lib)
target_compile_options(${PROJECT_NAME} PRIVATE -fPIC -O3 -flto -mno-avx2)
target_link_libraries(${PROJECT_NAME} PRIVATE raisim::raisim)
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC dependencies/networks_minimal/include
         dependencies/actuation_dynamics/include ${EIGEN3_INCLUDE_DIRS})

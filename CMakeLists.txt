cmake_minimum_required(VERSION 3.18)

project(partools LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add options
option(USING_CUDA "Enable CUDA support" OFF)
option(USING_OMP "Enable OpenMP support" OFF)
option(USING_HIP "Enable HIP support" OFF)

if(USING_CUDA AND USING_HIP)
  message(FATAL_ERROR "CUDA and HIP support cannot be enabled at the same time.")
endif()

# Create interface library
add_library(partools INTERFACE)

# Install header and directory
install(FILES ${CMAKE_SOURCE_DIR}/partools.hpp DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/partools DESTINATION include)

# Handle options
if(USING_OMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(partools INTERFACE OpenMP::OpenMP_CXX)
endif()

if(USING_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(partools INTERFACE CUDA::cudart)
  target_compile_options(partools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

  if (USING_OMP)
    target_compile_options(partools INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif()
endif()

if(USING_HIP)
  find_package(HIP REQUIRED)
  target_link_libraries(partools INTERFACE HIP::hiprtc)
endif()

# Add compile options
target_compile_options(partools INTERFACE -Wall)

enable_testing()

# Add tests for all .cpp files in the test directory
file(GLOB TEST_SOURCES ${CMAKE_SOURCE_DIR}/test/*.cpp)

foreach(TEST_SOURCE ${TEST_SOURCES})
  get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)

  if(USING_CUDA)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE CUDA)
  else()
    add_executable(${TEST_NAME} ${TEST_SOURCE})
  endif()
  
  set_target_properties(${TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${TEST_NAME} PRIVATE partools)
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_BINARY_DIR}/test/${TEST_NAME})
endforeach()
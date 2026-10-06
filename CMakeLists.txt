cmake_minimum_required(VERSION 3.16)
project(theft_guard CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build with optimizations unless told otherwise.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Every program uses the same settings: headers from src/, OpenCV, threads.
function(add_program name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${name} PRIVATE ${OpenCV_LIBS} Threads::Threads)
  target_compile_options(${name} PRIVATE -Wall -Wextra
    $<$<CONFIG:Release>:-O2 -march=native>)
endfunction()

# Main system
add_program(guard src/guard.cpp)

# Benchmarks
add_program(bench_single  bench/bench_single.cpp)
add_program(bench_threads bench/bench_threads.cpp)
add_program(bench_multi   bench/bench_multi.cpp)

# Step-by-step practice programs
foreach(name hello camera model_test detect track multi_view multi_detect)
  add_program(${name} experiments/${name}.cpp)
endforeach()

cmake_minimum_required(VERSION 3.10)
project(fire_detect)

set(TORCH_LIBRARIES torch_cpu)
set(Torch_DIR /opt/libtorch/share/cmake/Torch)


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

add_executable(
    fire_detect
    main.cpp
    src/detection.cpp
)

target_link_libraries(
    fire_detect
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
)
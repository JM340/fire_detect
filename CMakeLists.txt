cmake_minimum_required(VERSION 3.10)
project(fire_detect)

if(NOT DEFINED OrbbecSDK_FOUND)
    set(OrbbecSDK_DIR "/opt/OrbbecSDK/lib" )
    message(STATUS "OrbbecSDK_DIR: ${OrbbecSDK_DIR}")
    find_package(OrbbecSDK REQUIRED)
endif()

set(TORCH_LIBRARIES torch_cpu)
set(Torch_DIR /home/robot/pytorch/torch/share/cmake/Torch)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(SFML REQUIRED COMPONENTS audio)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${OrbbecSDK_INCLUDE_DIRS}
)

add_executable(
    fire_detect
    main.cpp
    src/detection.cpp
    src/camera.cpp
)

target_link_libraries(
    fire_detect
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${OrbbecSDK_LIBS}
    sfml-audio
)
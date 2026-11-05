cmake_minimum_required(VERSION 3.20)
project(waferwise_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(waferwise_infer src/main.cpp)
target_link_libraries(waferwise_infer PRIVATE openvino::runtime ${OpenCV_LIBS})

if(MSVC)
  target_compile_options(waferwise_infer PRIVATE /W4)
else()
  target_compile_options(waferwise_infer PRIVATE -Wall -Wextra)
endif()

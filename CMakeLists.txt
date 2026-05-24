cmake_minimum_required(VERSION 3.16)
project(content-based-image-retrieval CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_custom_target(copy_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
)

find_package(OpenCV REQUIRED)

add_executable(main
    ./src/main.cpp
    ./src/matching.cpp
)
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
)
target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# Build the 1 kHz Kortex UDP bridge.
#
# The Kortex C++ API is not vendored. Point KORTEX_DIR at the extracted API
# (the directory that contains include/ and lib/release/libKortexApiCpp.a):
#
#   cmake -S kinova_lfd/envs/base/kortex_udp/cpp -B kinova_lfd/envs/base/kortex_udp/cpp/build \
#       -DKORTEX_DIR=/path/to/kortex_api
#   cmake --build kinova_lfd/envs/base/kortex_udp/cpp/build

cmake_minimum_required(VERSION 3.16)
project(kortex_udp_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT KORTEX_DIR)
    message(FATAL_ERROR
        "KORTEX_DIR is not set. Pass -DKORTEX_DIR= to the Kortex C++ API root.")
endif()

set(KORTEX_LIB "${KORTEX_DIR}/lib/release/libKortexApiCpp.a")
if(NOT EXISTS "${KORTEX_LIB}")
    message(FATAL_ERROR "Kortex static library not found: ${KORTEX_LIB}")
endif()

add_executable(kortex_udp_bridge
    kortex_udp_bridge.cpp
    controller.cpp
)

target_compile_definitions(kortex_udp_bridge PRIVATE _OS_UNIX)
target_include_directories(kortex_udp_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(kortex_udp_bridge SYSTEM PRIVATE
    ${KORTEX_DIR}/include
    ${KORTEX_DIR}/include/client
    ${KORTEX_DIR}/include/client_stubs
    ${KORTEX_DIR}/include/common
    ${KORTEX_DIR}/include/messages
)
target_compile_options(kortex_udp_bridge PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(kortex_udp_bridge PRIVATE ${KORTEX_LIB} pthread dl)

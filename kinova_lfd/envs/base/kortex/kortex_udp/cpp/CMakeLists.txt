# Build the 1 kHz Kortex UDP bridge.
#
# The Kortex C++ API is not vendored. Point KORTEX_DIR at the extracted API
# (the directory that contains include/ and lib/release/libKortexApiCpp.a):
#
#   cmake -S kinova_lfd/envs/base/kortex/kortex_udp/cpp \
#       -B kinova_lfd/envs/base/kortex/kortex_udp/cpp/build \
#       -DKORTEX_DIR=/path/to/kortex_api
#   cmake --build kinova_lfd/envs/base/kortex/kortex_udp/cpp/build

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
    controller/controller.cpp
    controller/gen3_model.cpp
    controller/joint_trajectory_controller/joint_trajectory_controller.cpp
    controller/twist_controller/servo_config.cpp
    controller/twist_controller/twist_controller.cpp
)

target_compile_definitions(kortex_udp_bridge PRIVATE
    _OS_UNIX
    BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
    BOOST_MPL_LIMIT_LIST_SIZE=30
    BOOST_MPL_LIMIT_VECTOR_SIZE=30
    BOOST_FUSION_INVOKE_MAX_ARITY=12
)
target_include_directories(kortex_udp_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(kortex_udp_bridge SYSTEM PRIVATE
    ${KORTEX_DIR}/include
    ${KORTEX_DIR}/include/client
    ${KORTEX_DIR}/include/client_stubs
    ${KORTEX_DIR}/include/common
    ${KORTEX_DIR}/include/messages
    /opt/ros/jazzy/include
    /usr/include/eigen3
)
target_compile_options(kortex_udp_bridge PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_directories(kortex_udp_bridge PRIVATE /opt/ros/jazzy/lib/x86_64-linux-gnu)
target_link_libraries(kortex_udp_bridge PRIVATE ${KORTEX_LIB} pinocchio_parsers pinocchio_default pthread dl)
set_target_properties(kortex_udp_bridge PROPERTIES
    BUILD_RPATH "/opt/ros/jazzy/lib/x86_64-linux-gnu"
)

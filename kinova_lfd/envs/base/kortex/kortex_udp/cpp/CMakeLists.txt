# Build the 1 kHz Kortex UDP bridge.
#
# Dependencies are installed on their own. This file does not look in a ROS 2
# workspace. Pinocchio 4, Eigen, and Boost come from the system or from
# -DPINOCCHIO_DIR. The Kortex C++ API is the directory that contains include/
# and lib/release/libKortexApiCpp.a.
#
#   cmake -S kinova_lfd/envs/base/kortex/kortex_udp/cpp \
#       -B kinova_lfd/envs/base/kortex/kortex_udp/cpp/build \
#       -DKORTEX_DIR="$PWD/kinova_lfd/envs/base/kortex/kortex_cpp_api" \
#       -DPINOCCHIO_DIR=/usr/local
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

set(PINOCCHIO_DIR "" CACHE PATH
    "Prefix of a standalone Pinocchio 3 install. Leave empty to search /usr and /usr/local.")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost REQUIRED)

find_path(PINOCCHIO_INCLUDE_DIR
    NAMES pinocchio/parsers/urdf.hpp
    HINTS ${PINOCCHIO_DIR}
    PATH_SUFFIXES include
)
find_library(PINOCCHIO_PARSERS_LIB
    NAMES pinocchio_parsers
    HINTS ${PINOCCHIO_DIR}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu lib64
)
find_library(PINOCCHIO_DEFAULT_LIB
    NAMES pinocchio_default
    HINTS ${PINOCCHIO_DIR}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu lib64
)
if(NOT PINOCCHIO_INCLUDE_DIR OR NOT PINOCCHIO_PARSERS_LIB OR NOT PINOCCHIO_DEFAULT_LIB)
    message(FATAL_ERROR
        "Pinocchio 4 was not found. Install it outside ROS (see the top-level README) "
        "and pass -DPINOCCHIO_DIR= to its prefix if it is not under /usr or /usr/local.")
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
    ${PINOCCHIO_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)
target_compile_options(kortex_udp_bridge PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(kortex_udp_bridge PRIVATE
    ${KORTEX_LIB}
    ${PINOCCHIO_PARSERS_LIB}
    ${PINOCCHIO_DEFAULT_LIB}
    Eigen3::Eigen
    pthread
    dl
)

get_filename_component(PINOCCHIO_LIB_DIR "${PINOCCHIO_DEFAULT_LIB}" DIRECTORY)
set_target_properties(kortex_udp_bridge PROPERTIES
    BUILD_RPATH "${PINOCCHIO_LIB_DIR}"
)

cmake_minimum_required(VERSION 3.10)
project(HSMDIF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(Protobuf REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp)

# Generate gRPC files
set(PROTO_FILES inference_service.proto)
set(PROTO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROTO_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    list(APPEND PROTO_SRCS "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.cc")
    list(APPEND PROTO_HDRS "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.h")
    list(APPEND GRPC_SRCS "${PROTO_OUT_DIR}/${PROTO_NAME}.grpc.pb.cc")
    list(APPEND GRPC_HDRS "${PROTO_OUT_DIR}/${PROTO_NAME}.grpc.pb.h")

    add_custom_command(
        OUTPUT "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.cc"
               "${PROTO_OUT_DIR}/${PROTO_NAME}.pb.h"
               "${PROTO_OUT_DIR}/${PROTO_NAME}.grpc.pb.cc"
               "${PROTO_OUT_DIR}/${PROTO_NAME}.grpc.pb.h"
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${PROTO_OUT_DIR}
             --grpc_out=${PROTO_OUT_DIR}
             --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
             -I ${PROTO_SRC_DIR}
             ${PROTO_SRC_DIR}/${PROTO_FILE}
        DEPENDS ${PROTO_SRC_DIR}/${PROTO_FILE}
        COMMENT "Generating gRPC files for ${PROTO_FILE}"
    )
endforeach()

# Build llama.cpp as a static library
add_library(llama STATIC
    llama.cpp/llama.cpp
    llama.cpp/ggml.c
    # Add other necessary source files from llama.cpp
)
target_include_directories(llama PUBLIC llama.cpp)

# Build the main executable
add_executable(hsmdif
    src/hsmdif.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

target_include_directories(hsmdif PRIVATE ${PROTO_OUT_DIR})

target_link_libraries(hsmdif
    llama
    Boost::program_options
    nlohmann_json::nlohmann_json
    gRPC::grpc++_reflection
    gRPC::grpc++
    ${Protobuf_LIBRARIES}
)
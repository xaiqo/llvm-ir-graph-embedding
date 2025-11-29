cmake_minimum_required(VERSION 3.20)

# Project Definition
project(LLVMGraphExtractor 
    VERSION 2.1.0 
    DESCRIPTION "LLVM IR to TOON (Token-Oriented Object Notation) Extractor"
    LANGUAGES CXX C
)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# 1. LLVM 17 (Strict Requirement)
find_package(LLVM 17 REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Add -fvisibility=default so LLVM can find the plugin entry point
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fPIC -Wall -O3 -fvisibility=default")

# -----------------------------------------------------------------------------
# Build Targets
# -----------------------------------------------------------------------------

file(GLOB_RECURSE PASS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(NOT PASS_SOURCES)
    message(FATAL_ERROR "No C++ source files found in ${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()

add_library(GraphExtractor SHARED ${PASS_SOURCES})

llvm_map_components_to_libnames(llvm_libs 
    Support 
    Core 
    IRReader 
    Analysis 
    Passes 
)

target_link_libraries(GraphExtractor 
    PRIVATE
)

set_target_properties(GraphExtractor PROPERTIES 
    LINK_FLAGS "-Wl,--no-undefined" # Actually, let's try enforcing resolution to find what's missing
)
find_library(LLVM_SHARED_LIB NAMES LLVM-17 LLVM)
if(LLVM_SHARED_LIB)
    message(STATUS "Linking against shared LLVM: ${LLVM_SHARED_LIB}")
    target_link_libraries(GraphExtractor PRIVATE ${LLVM_SHARED_LIB})
else()
    set_target_properties(GraphExtractor PROPERTIES LINK_FLAGS "-Wl,-undefined,dynamic_lookup")
endif()

# Output configuration
set_target_properties(GraphExtractor PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

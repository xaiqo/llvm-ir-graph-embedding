cmake_minimum_required(VERSION 3.20)

# Project Definition
project(LLVMGraphExtractor 
    VERSION 2.1.0 
    DESCRIPTION "LLVM IR to TOON (Token-Oriented Object Notation) Extractor"
    LANGUAGES CXX
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -fPIC -Wall -O3")

# -----------------------------------------------------------------------------
# Build Targets
# -----------------------------------------------------------------------------

file(GLOB_RECURSE PASS_SOURCES "src/*.cpp")

add_library(GraphExtractor MODULE ${PASS_SOURCES})

# Map LLVM components (New Pass Manager requires 'Passes')
llvm_map_components_to_libnames(llvm_libs 
    Support 
    Core 
    IRReader 
    Analysis 
    Passes 
)

# Link libraries
target_link_libraries(GraphExtractor 
    PUBLIC 
    ${llvm_libs}
)

# Output configuration
set_target_properties(GraphExtractor PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

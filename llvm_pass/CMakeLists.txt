cmake_minimum_required(VERSION 3.20)

# Project Definition
project(LLVMGraphExtractor 
    VERSION 1.0.0 
    DESCRIPTION "High-performance LLVM IR to Graph Extractor for Neuro-Symbolic AI"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Dependencies & Configuration
# -----------------------------------------------------------------------------

# Enforce LLVM 17 as per system specification
find_package(LLVM 17 REQUIRED CONFIG)

message(STATUS "---------------------------------------------------")
message(STATUS "Build Configuration:")
message(STATUS "  LLVM Version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "  LLVM Dir:     ${LLVM_DIR}")
message(STATUS "  Source Dir:   ${CMAKE_SOURCE_DIR}")
message(STATUS "---------------------------------------------------")

# Add LLVM include directories
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# -----------------------------------------------------------------------------
# Compiler Flags (Production Grade)
# -----------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LLVM requires RTTI to be disabled.
# We also add -Wall -Wextra to ensure code quality standard.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -Wall -Wextra -Wno-unused-parameter")

# -----------------------------------------------------------------------------
# Target Definition
# -----------------------------------------------------------------------------

# Locate source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Define the shared library (plugin)
add_library(GraphExtractor MODULE ${SOURCES})

# Map LLVM components to their libraries
# We need:
# - Core: Basic IR manipulation
# - Support: Utilities
# - Analysis: Alias Analysis (Memory Dependence)
llvm_map_components_to_libnames(llvm_libs Support Core IRReader Analysis)

# Link libraries
target_link_libraries(GraphExtractor PUBLIC ${llvm_libs})

# -----------------------------------------------------------------------------
# Output Configuration
# -----------------------------------------------------------------------------

# Ensure the plugin is named 'GraphExtractor.so' without the 'lib' prefix
# This simplifies loading it via the 'opt' tool.
set_target_properties(GraphExtractor PROPERTIES 
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# libxsmm
include(xsmm)
message (STATUS "LIBXSMM Include dir: ${XSMM_INCLUDE_DIRS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonCPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc LINK_LIBS TritonCPUToLLVM TritonCPUTransforms TritonCPUXsmm)
  target_link_libraries(TritonCPU PUBLIC MLIRVectorToSCF MLIRAffineToStandard MLIRMathToLibm MLIRAMXToLLVMIRTranslation MLIRMemRefTransforms)
endif()

add_library(TritonCPURuntime SHARED ${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpu_runtime.cpp)
target_link_libraries(TritonCPURuntime PRIVATE LLVMSupport)

add_library(TritonCPUXsmmRuntime SHARED ${CMAKE_CURRENT_SOURCE_DIR}/lib/Xsmm/runtime/XsmmRunnerUtils.cpp)
target_link_libraries(TritonCPUXsmmRuntime PRIVATE xsmm)
set_property(TARGET TritonCPUXsmmRuntime PROPERTY CXX_STANDARD 11)
target_compile_definitions(TritonCPUXsmmRuntime PRIVATE mlir_c_runner_utils_EXPORTS)
target_include_directories(TritonCPUXsmmRuntime
  PUBLIC
    $<BUILD_INTERFACE:${XSMM_INCLUDE_DIRS}>
)

# Build and link sleef
set(SLEEF_BUILD_SHARED_LIBS ON CACHE BOOL "Build sleef shared lib" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will not be built." FORCE)
add_subdirectory("${CMAKE_SOURCE_DIR}/third_party/sleef" sleef)
# Override sleef's output directory with our own
set_target_properties(sleef PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

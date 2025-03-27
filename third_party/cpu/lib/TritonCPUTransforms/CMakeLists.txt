add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    ConvertDotOp/ConvertDotOpToUkernelOps.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)

if (dnnl_FOUND)
  target_link_libraries(TritonCPUTransforms PRIVATE DNNL::dnnl)
endif()

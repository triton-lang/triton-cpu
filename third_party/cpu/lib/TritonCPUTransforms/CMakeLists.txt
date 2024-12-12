add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)

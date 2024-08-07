add_triton_library(TritonCPUTransforms
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    ConvertDotProduct.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)

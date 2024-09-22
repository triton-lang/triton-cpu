#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-warps")).getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if (!mod->hasAttr("triton_gpu.num-ctas"))
        return 1;
      return cast<IntegerAttr>(mod->getAttr("triton_gpu.num-ctas")).getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "triton_gpu.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu.threads-per-warp");
      if(!threadsPerWarp) {
        return isCPUMode(mod) ? 1 : 32;
      }
      return cast<IntegerAttr>(threadsPerWarp).getInt();
    }

    static bool isCPUMode(ModuleOp mod) {
      Attribute cpuMode = mod->getDiscardableAttr("triton_gpu.cpu-mode");
      if(!cpuMode) {
        return false;
      }
      return cast<BoolAttr>(cpuMode).getValue();
    }
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif

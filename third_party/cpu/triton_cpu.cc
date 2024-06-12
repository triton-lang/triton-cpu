#include "TritonCPUToLLVM/Passes.h"
#include "TritonToTritonCPU/Passes.h"

#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonCPUToLLVM/Passes.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace py = pybind11;

void init_triton_cpu_passes_ttcpuir(py::module &&m) {
  using namespace mlir::triton;
  // m.def("add_to_llvmir", [](mlir::PassManager &pm) {
  //   pm.addPass(mlir::triton::createConvertTritonCPUToLLVMPass());
  // });
  m.def("add_triton_to_triton_cpu_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonToTritonCPUPipelineBuilder(pm);
  });
  m.def("add_triton_cpu_to_llvmir_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::cpu::tritonCPUToLLVMPipelineBuilder(pm);
  });
  m.def("add_vector_to_scf", [](mlir::PassManager &pm, bool full_unroll,
                                unsigned target_rank, bool lower_tensors) {
    mlir::VectorTransferToSCFOptions opts;
    opts.setTargetRank(target_rank);
    opts.enableFullUnroll(full_unroll);
    opts.enableLowerTensors(lower_tensors);
    pm.addPass(mlir::createConvertVectorToSCFPass(opts));
  });
  m.def("add_lower_vector_multi_dim", [](mlir::PassManager &pm) {
    pm.addNestedPass<mlir::triton::FuncOp>(
        mlir::triton::cpu::createLowerMultiReductionPass());
  });
  m.def("add_vector_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertVectorToLLVMPass());
  });
  m.def("add_lower_affine", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createLowerAffinePass());
  });
  m.def("add_memref_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  });
  m.def("add_math_to_libm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToLibmPass());
  });
  m.def("add_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  });
}

bool getBoolEnv(const std::string &env) {
  const char *s = std::getenv(env.c_str());
  std::string str(s ? s : "");
  std::transform(str.begin(), str.end(), str.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return (str == "on" || str == "true" || str == "1");
}

llvm::orc::ThreadSafeContext &getThreadSafeContext() {
  static llvm::orc::ThreadSafeContext tsc;
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {
    auto context = std::make_unique<llvm::LLVMContext>();
    tsc = llvm::orc::ThreadSafeContext(std::move(context));
  });
  return tsc;
}

std::string llvmErrToString(const llvm::Error &err) {
  std::string res;
  llvm::raw_string_ostream os(res);
  os << err;
  return res;
};

struct CompiledKernel {
  std::unique_ptr<llvm::orc::ExecutionSession> execution_session;
  std::unique_ptr<llvm::DataLayout> data_layout;
  std::unique_ptr<llvm::orc::MangleAndInterner> mangle;
  std::unique_ptr<llvm::orc::RTDyldObjectLinkingLayer> object_layer;
  std::unique_ptr<llvm::orc::IRCompileLayer> compiler_layer;
  llvm::orc::JITDylib *dylib = nullptr;

  CompiledKernel() = default;
  CompiledKernel(CompiledKernel &&) = default;

  ~CompiledKernel() {
    if (execution_session)
      llvm::cantFail(execution_session->endSession());
  }
};

std::vector<std::unique_ptr<CompiledKernel>> compiled_kernels;

void init_triton_cpu_utils(py::module &&m) {
  using namespace mlir::triton;
  m.def("load_binary", [](std::string name, std::string llvmBC, int shared, int devId) {
    auto res = py::object(py::cast(nullptr));
    
    llvm::LLVMContext context;
    auto buf = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(llvmBC.c_str(), llvmBC.length()));

    auto mod = llvm::parseBitcodeFile(*buf, context);
    if (!mod) {
      std::cerr << "Failed to parse LLVM bitcode module" << std::endl;
      return res;
    }
  
    if (getBoolEnv("MLIR_ENABLE_DUMP")) {
      llvm::errs() << "********** Loaded Module (kernel_name=" << name
                   << ") **********\n"
                   << **mod << "\n";
    }
  
    auto init_err = llvm::InitializeNativeTarget();
    if (init_err) {
      std::cerr << "Failed to initialize native target." << std::endl;
      return res;
    }
  
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  
    auto self_epc =
        llvm::cantFail(llvm::orc::SelfExecutorProcessControl::Create());
  
    auto detect_host_res = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!detect_host_res) {
      std::cerr << "Failed to initialize JITTargetMachineBuilder: "
                << llvmErrToString(detect_host_res.takeError());
      return res;
    }
    llvm::orc::JITTargetMachineBuilder tmb = std::move(*detect_host_res);
  
    auto data_layout_res = tmb.getDefaultDataLayoutForTarget();
    
    std::cout << "cpu: " << tmb.getCPU() << std::endl;
    std::cout << "target triple: " << tmb.getTargetTriple().getTriple() << std::endl;

    if (!data_layout_res) {
      std::cerr << "Failed to initialize data layout: "
                << llvmErrToString(data_layout_res.takeError());
      return res;
    }
  
    CompiledKernel kernel;
    kernel.execution_session =
        std::make_unique<llvm::orc::ExecutionSession>(std::move(self_epc));
    kernel.data_layout =
        std::make_unique<llvm::DataLayout>(std::move(*data_layout_res));
    kernel.mangle = std::make_unique<llvm::orc::MangleAndInterner>(
        *kernel.execution_session, *kernel.data_layout);
    kernel.object_layer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
        *kernel.execution_session,
        []() { return std::make_unique<llvm::SectionMemoryManager>(); });
    kernel.compiler_layer = std::make_unique<llvm::orc::IRCompileLayer>(
        *kernel.execution_session, *kernel.object_layer,
        std::make_unique<llvm::orc::ConcurrentIRCompiler>(std::move(tmb)));
  
    auto dylib_res = kernel.execution_session->createJITDylib("<main>");
    if (!dylib_res) {
      std::cerr << "Failed to create initialize JITDylib: "
                << llvmErrToString(dylib_res.takeError());
      return res;
    }
  
    kernel.dylib = &(*dylib_res);
    kernel.dylib->addGenerator(llvm::cantFail(
        llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            kernel.data_layout->getGlobalPrefix())));
    
    // Compile module.
    (**mod).setDataLayout(*kernel.data_layout);
    llvm::orc::ThreadSafeModule tsm(std::move(*mod), getThreadSafeContext());
    auto err = kernel.compiler_layer->add(*kernel.dylib, std::move(tsm));
    if (err) {
      std::cerr << "Cannot add LLVM module: " << llvmErrToString(err);
      return res;
    }
  
    // Find kernel function pointer.
    auto lookup_res =
        kernel.execution_session->lookup({kernel.dylib}, (*kernel.mangle)(name));
    if (!lookup_res) {
      std::cerr << "Failed to find function " << std::string(name)
                << "\nError: " << llvmErrToString(lookup_res.takeError());
      return res;
    }
    uint64_t fn_ptr = lookup_res->getAddress().getValue();

    compiled_kernels.push_back(
        std::make_unique<CompiledKernel>(std::move(kernel)));
    auto *kernel_ptr = compiled_kernels.back().get();
  
    return py::object(py::make_tuple(reinterpret_cast<uint64_t>(kernel_ptr),
                         reinterpret_cast<uint64_t>(fn_ptr), 0, 0));
  });
}  

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_cpu_passes_ttcpuir(passes.def_submodule("ttcpuir"));
  init_triton_cpu_utils(m.def_submodule("utils"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::cpu::TritonCPUDialect,
                    mlir::vector::VectorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("find_kernel_names", [](mlir::ModuleOp &mod) {
    std::vector<std::string> res;
    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      if (funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}

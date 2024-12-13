// #include <cpu/x64/amx_tile_configure.hpp>
// #include <cpu/x64/brgemm/brgemm.hpp>
// #include <cpu/x64/brgemm/brgemm_types.hpp>
// #include <cpu/x64/cpu_isa_traits.hpp>

#include <oneapi/dnnl/dnnl_types.h>
#include <oneapi/dnnl/dnnl_ukernel.hpp>
#include <oneapi/dnnl/dnnl_ukernel_types.h>

#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

// using namespace dnnl::impl::cpu::x64;
using namespace dnnl;
using namespace dnnl::ukernel;

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {
// dummy definition for DNNL lite linkage
__attribute__((weak)) void print_verbose_header() {}
} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl

static constexpr int PALETTE_SIZE = 64;
static constexpr int DEFAULT_KERNEL_SIZE = 1024;
static constexpr int MAX_KERNEL_SIZE = 2048;

using read_lock_guard_t = std::shared_lock<std::shared_mutex>;
using write_lock_guard_t = std::unique_lock<std::shared_mutex>;
static std::shared_mutex g_brgemm_lock;

// TODO(haixin): use syscall to determine page size?
static constexpr size_t SCRATCH_SIZE = 2 * 4096;
// TODO(haixin): need to use custom thread management for scratch in the future?
static thread_local char scratch[SCRATCH_SIZE] = {0};

namespace {
template <typename T> struct RawMemRefDescriptor {
  const T *allocated;
  const T *aligned;
  intptr_t offset;
  intptr_t sizesAndStrides[];
};
} // namespace

extern "C" {

EXPORT void *create_brgemm_ukernel(int64_t M, int64_t N, int64_t K_k,
                                   int64_t batch_size, int64_t lda, int64_t ldb,
                                   int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                                   int64_t dtypeC) {
  using K = std::array<int64_t, 10>;
  std::cout << "Args: M - " << M << ", N - " << N << ", K - " << K_k
            << ", bath - " << batch_size << ", lda - " << lda << ", ldb - "
            << ldb << ", ldc - " << ldc << ", dtype a - " << dtypeA
            << ", dtype b - " << dtypeB << ", dtype c - " << dtypeC << "\n";
  K key{M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC};

  static std::map<K, dnnl::ukernel::brgemm> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernels.count(key) != 0) {
      return &savedUkernels.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernels.count(key) != 0) {
    return &savedUkernels.find(key)->second;
  }

  auto dnnl_dtypeA = static_cast<dnnl::memory::data_type>(dtypeA);
  auto dnnl_dtypeB = static_cast<dnnl::memory::data_type>(dtypeB);
  auto dnnl_dtypeC = static_cast<dnnl::memory::data_type>(dtypeC);

  dnnl::ukernel::brgemm brg;
  brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, ldb, ldc, dnnl_dtypeA,
                              dnnl_dtypeB, dnnl_dtypeC);

  // Instruct the kernel to append the result to C tensor.
  brg.set_add_C(true);
  // Finalize the initialization.
  brg.finalize();
  // Generate the executable JIT code for the objects.
  brg.generate();

  auto it = savedUkernels.insert({key, brg});
  return &it.first->second;
}

EXPORT void *create_transform_ukernel(int64_t K, int64_t N, int64_t in_ld,
                                      int64_t out_ld, int64_t inDtype,
                                      int64_t outDtype) {
  using K_t = std::array<int64_t, 6>;
  K_t key{K, N, in_ld, out_ld, inDtype, outDtype};

  static std::map<K_t, dnnl::ukernel::transform> savedUkernels;
  {
    read_lock_guard_t r_g(g_brgemm_lock);
    if (savedUkernels.count(key) != 0) {
      return &savedUkernels.find(key)->second;
    }
  }

  write_lock_guard_t w_g(g_brgemm_lock);

  if (savedUkernels.count(key) != 0) {
    return &savedUkernels.find(key)->second;
  }

  // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
  // special VNNI format for low precision data types, e.g., bfloat16_t.
  // Note: the routine doesn't provide a `batch_size` argument in the
  // constructor as it can be either incorporated into `K` dimension, or
  // manually iterated over in a for-loop on the user side.
  dnnl::ukernel::transform pack_B(
      /* K = */ K, /* N = */ N,
      /* in_pack_type = */ pack_type::no_trans,
      /* in_ld = */ in_ld,
      /* out_ld = */ out_ld,
      /* in_dt = */ static_cast<dnnl::memory::data_type>(inDtype),
      /* out_dt = */ static_cast<dnnl::memory::data_type>(outDtype));

  // Pack B routine execution.
  // Note: usually should be split to process only that part of B that the
  // ukernel will execute.
  pack_B.generate();

  auto it = savedUkernels.insert({key, pack_B});
  return &it.first->second;
}

EXPORT void call_all(const void *transform_k, const void *brg_k, void *A_ptr,
                     void *original_B_ptr, void *C_ptr, void *scratchpad,
                     int64_t A_step_in_bytes, int64_t B_step_in_bytes,
                     int64_t B_block_size_in_bytes, int64_t num_batches,
                     bool skip_packing = false) {

  uint8_t *blocked_data = (uint8_t *)original_B_ptr;
  uint8_t *B_ptr_calc = (uint8_t *)original_B_ptr;
  std::cout << "Call Transform: " << transform_k << " Brg: " << brg_k
            << ", a: " << A_ptr << ", b: " << original_B_ptr << ", c: " << C_ptr
            << ", scr: " << scratchpad << "\n";
  std::cout << "steps: " << A_step_in_bytes << " " << B_step_in_bytes << " "
            << B_block_size_in_bytes << " n: " << num_batches << "\n";

  auto pack_B = reinterpret_cast<const dnnl::ukernel::transform *>(transform_k);
  auto brg = reinterpret_cast<const dnnl::ukernel::brgemm *>(brg_k);
  std::cout << " vanilla check pack: "
            << ((brg->get_B_pack_type() == pack_type::pack32) ? "true"
                                                              : "false")
            << "\n";
  bool need_packing =
      brg->get_B_pack_type() == pack_type::pack32 && !skip_packing;
  if (need_packing) {
    std::cout << "Will be packed. \n";
    // output

    // blocked_B_size = block_K * block_n * dtype; // ldb * K_k *
    // memory::data_type_size(b_dt);

    blocked_data = new uint8_t[B_block_size_in_bytes * num_batches];

    pack_B->execute(original_B_ptr, blocked_data);
  }

  brg->set_hw_context();

  if (need_packing) {
    std::cout << "[packed]  b_ptr_calc: " << (void *)B_ptr_calc
              << " blocked_ptr: " << (void *)blocked_data << "\n";
  } else {
    std::cout << "[unpacked]  b_ptr_calc: " << (void *)B_ptr_calc
              << " blocked_ptr: " << (void *)blocked_data << "\n";
  }

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i =
        i * A_step_in_bytes; // * a_dt_size; // K_k * a_dt_size;

    memory::dim B_offset_i;
    if (need_packing) {
      pack_B->execute(B_ptr_calc + i * B_step_in_bytes,
                      blocked_data + i * B_block_size_in_bytes);
      B_offset_i = i * B_block_size_in_bytes;
    } else {
      B_offset_i = i * B_step_in_bytes;
    }
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }

  size_t scratchpad_size = brg->get_scratchpad_size();
  std::vector<uint8_t> scratchpad_sm(scratchpad_size);

  //  An execute call. `A_B` is a vector of pointers to A and packed B
  //  tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg->execute(A_ptr, blocked_data, A_B_offsets, C_ptr, scratchpad_sm.data());

  dnnl::ukernel::brgemm::release_hw_context();

  if (need_packing) {
    delete blocked_data;
  };
}

EXPORT void call_transform(const void *transform_k, const void *original_data,
                           void *blocked_data) {
  auto pack_B = reinterpret_cast<const dnnl::ukernel::transform *>(transform_k);
  pack_B->execute(original_data, blocked_data);
}

// Most questionable function - no shure where to leave forming of offsets lists
// maybe too difficult in client code
// void prepare_buffers(int64_t batch_size) {
//   // BRGeMM ukernel execute section.
//   // Prepare buffers for execution.
//   std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(batch_size);
//   for (memory::dim i = 0; i < batch_size; i++) {
//     const memory::dim A_offset_i = i * K_k * a_dt_size;
//     const memory::dim B_offset_i =
//         need_pack ? i * blocked_B_size : i * N * K_k * b_dt_size;
//     A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
//   }
// }

// for perf targets
EXPORT void prepare_hw_context(const void *brg_k) {
  auto brg = reinterpret_cast<const dnnl::ukernel::brgemm *>(brg_k);
  brg->set_hw_context();
}

EXPORT void call_brgemm(const void *brg_k, void *A_ptr, void *B_ptr,
                        void *C_ptr, void *scratchpad, int64_t A_step_in_bytes,
                        int64_t B_step_in_bytes, int64_t num_batches) {
  std::cout << "Call Brg: " << brg_k << ", " << A_ptr << ", " << B_ptr << ", "
            << C_ptr << ", " << scratchpad << "\n";
  std::cout << "steps: " << A_step_in_bytes << " " << B_step_in_bytes
            << " n: " << num_batches << "\n";

  if (A_ptr == nullptr || B_ptr == nullptr || C_ptr == nullptr) {
    std::cout << "----------------FAIL----------------\n";
    return;
  }

  // auto dnnl_dtypeA = static_cast<dnnl::memory::data_type>(dtypeA);
  // auto dnnl_dtypeB = static_cast<dnnl::memory::data_type>(dtypeB);

  // const size_t a_dt_size = memory::data_type_size(dnnl_dtypeA);
  // const size_t b_dt_size = memory::data_type_size(dnnl_dtypeB);

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i =
        i * A_step_in_bytes; // * a_dt_size; // K_k * a_dt_size;
    const memory::dim B_offset_i =
        i * B_step_in_bytes; // * b_dt_size; // N * K_k * b_dt_size;
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }

  auto brg = reinterpret_cast<const dnnl::ukernel::brgemm *>(brg_k);

  size_t scratchpad_size = brg->get_scratchpad_size();
  std::vector<float> scratchpad_sm(scratchpad_size);
  // std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(3);
  //  An execute call. `A_B` is a vector of pointers to A and packed B
  //  tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg->execute(A_ptr, B_ptr, A_B_offsets, C_ptr, scratchpad_sm.data());
}

// at the end of execution
EXPORT void release_hw_context() {
  // Once all computations are done, need to release HW context.
  dnnl::ukernel::brgemm::release_hw_context();
}

} // extern C

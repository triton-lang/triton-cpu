#if defined(ONEDNN_AVAILABLE)
#include "oneapi/dnnl/dnnl_types.h"
#include "oneapi/dnnl/dnnl_ukernel.hpp"
#include "oneapi/dnnl/dnnl_ukernel_types.h"
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#error "DNNL Ukerenel ismissing"
#endif
#endif

#include <cassert>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>

#if defined(_MSC_VER)
#define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define EXPORT __attribute__((visibility("default")))
#else
#define EXPORT
#endif

#if defined(ONEDNN_AVAILABLE)
using namespace dnnl;
using namespace dnnl::ukernel;
#endif

using read_lock_guard_t = std::shared_lock<std::shared_mutex>;
using write_lock_guard_t = std::unique_lock<std::shared_mutex>;
static std::shared_mutex g_brgemm_lock;

extern "C" {

EXPORT void *create_brgemm_ukernel(int64_t M, int64_t N, int64_t K_k,
                                   int64_t batch_size, int64_t lda, int64_t ldb,
                                   int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                                   int64_t dtypeC) {
  using KeyT = std::array<int64_t, 10>;
  KeyT key{M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC};

  static std::map<KeyT, dnnl::ukernel::brgemm> savedUkernels;
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
  auto ret = &it.first->second;
  return ret;
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

void print_arr(uint8_t *ptr) {
  std::cout << "[";
  for (int i_b = 0; i_b < 100; i_b++) {
    std::cout << " " << unsigned(ptr[i_b]);
  }
  std::cout << "]\n";
}

EXPORT void call_all(const void *transform_k, const void *brg_k, void *A_ptr,
                     void *original_B_ptr, void *C_ptr, int64_t A_step_in_bytes,
                     int64_t B_step_in_bytes, int64_t B_block_size_in_bytes,
                     int64_t num_batches, bool skip_packing = false) {

  uint8_t *blocked_data = reinterpret_cast<uint8_t *>(original_B_ptr);
  const uint8_t *B_ptr_calc = reinterpret_cast<const uint8_t *>(original_B_ptr);

  auto pack_B = reinterpret_cast<const dnnl::ukernel::transform *>(transform_k);
  auto brg = reinterpret_cast<const dnnl::ukernel::brgemm *>(brg_k);

  bool need_packing =
      brg->get_B_pack_type() == pack_type::pack32 && !skip_packing;
  if (need_packing) {
    blocked_data = new uint8_t[B_block_size_in_bytes * num_batches];
  }

  brg->set_hw_context();

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i = i * A_step_in_bytes;

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
    delete[] blocked_data;
  };
}

} // extern C

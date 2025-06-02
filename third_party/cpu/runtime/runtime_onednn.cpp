#if defined(ONEDNN_AVAILABLE)
#include "oneapi/dnnl/dnnl_types.h"
#include "oneapi/dnnl/dnnl_ukernel.hpp"
#include "oneapi/dnnl/dnnl_ukernel_types.h"
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
#error "DNNL Ukerenel is missing"
#endif
#endif

#include <array>
#include <cassert>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <vector>

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

struct onednn_handle {
  dnnl::ukernel::transform transform;
  dnnl::ukernel::brgemm brg;
};

EXPORT void *create_brgemm(int64_t M, int64_t N, int64_t K_k,
                           int64_t batch_size, int64_t lda, int64_t ldb,
                           int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                           int64_t dtypeC, bool skip_packing) {
  using KeyT = std::array<int64_t, 10>;
  KeyT key{M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC};

  static std::map<KeyT, onednn_handle> savedUkernels;
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

  bool isPackingNeeded =
      dnnl::ukernel::brgemm::get_B_pack_type(dnnl_dtypeA, dnnl_dtypeB) ==
      dnnl::ukernel::pack_type::pack32;

  dnnl::ukernel::brgemm brg;
  if (!isPackingNeeded) {
    brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, ldb, ldc,
                                dnnl_dtypeA, dnnl_dtypeB, dnnl_dtypeC);
  } else {
    if (skip_packing) {
      // data is vnni prepacked
      // ldb is interleaved with vnni_factor
      // So we will use ldb divided to number of interleaved rows.
      // In general register size is 32 bits (4 bytes)
      // So number of interleaved is
      int vnni_pack_factor = 4 / dnnl::memory::data_type_size(dnnl_dtypeB);
      brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda,
                                  ldb / vnni_pack_factor, ldc, dnnl_dtypeA,
                                  dnnl_dtypeB, dnnl_dtypeC);
    } else {
      // Transform will be generated on this branch
      brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, N, ldc,
                                  dnnl_dtypeA, dnnl_dtypeB, dnnl_dtypeC);
    }
  }

  // Instruct the kernel to append the result to C tensor.
  brg.set_add_C(true);
  // Finalize the initialization.
  brg.finalize();

  // Generate the executable JIT code for the objects.
  brg.generate();

  dnnl::ukernel::transform tf;
  if (isPackingNeeded && !skip_packing) {
    // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
    // special VNNI format for low precision data types, e.g., bfloat16_t.
    // Note: the routine doesn't provide a `batch_size` argument in the
    // constructor as it can be either incorporated into `K` dimension, or
    // manually iterated over in a for-loop on the user side.
    dnnl::ukernel::transform pack_B(
        /* K = */ K_k, /* N = */ N, /* in_pack_type = */ pack_type::no_trans,
        /* in_ld = */ ldb, /* out_ld = */ N, /* in_dt = */ dnnl_dtypeB,
        /* out_dt = */ dnnl_dtypeB);

    pack_B.generate();
    tf = std::move(pack_B);
  }

  auto it = savedUkernels.insert({key, {tf, brg}});
  return &it.first->second;
}

EXPORT void brgemm_execute(const void *handle, void *A_ptr,
                           void *original_B_ptr, void *C_ptr,
                           int64_t A_step_in_bytes, int64_t B_step_in_bytes,
                           int64_t B_block_size_in_bytes, int64_t num_batches,
                           bool skip_packing) {

  uint8_t *blocked_data = reinterpret_cast<uint8_t *>(original_B_ptr);
  const uint8_t *B_ptr_calc = reinterpret_cast<const uint8_t *>(original_B_ptr);

  const onednn_handle *kernel = reinterpret_cast<const onednn_handle *>(handle);

  const auto pack_B = kernel->transform;
  const auto brg = kernel->brg;

  if (!skip_packing) {
    blocked_data = new uint8_t[B_block_size_in_bytes * num_batches];
  }

  brg.set_hw_context();

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i = i * A_step_in_bytes;

    memory::dim B_offset_i;
    if (!skip_packing) {
      pack_B.execute(B_ptr_calc + i * B_step_in_bytes,
                     blocked_data + i * B_block_size_in_bytes);
      B_offset_i = i * B_block_size_in_bytes;
    } else {
      B_offset_i = i * B_step_in_bytes;
    }
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }

  size_t scratchpad_size = brg.get_scratchpad_size();
  std::vector<uint8_t> scratchpad_sm(scratchpad_size);
  //  An execute call. `A_B` is a vector of pointers to A and packed B
  //  tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg.execute(A_ptr, blocked_data, A_B_offsets, C_ptr, scratchpad_sm.data());

  dnnl::ukernel::brgemm::release_hw_context();

  if (!skip_packing) {
    delete[] blocked_data;
  };
}

} // extern C

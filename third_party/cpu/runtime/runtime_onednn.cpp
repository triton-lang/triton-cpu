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

struct onednn_handle {
  dnnl::ukernel::transform transform;
  dnnl::ukernel::brgemm brg;
};

EXPORT void *create_brgemm(int64_t M, int64_t N, int64_t K_k,
                           int64_t batch_size, int64_t lda, int64_t ldb,
                           int64_t ldc, int64_t dtypeA, int64_t dtypeB,
                           int64_t dtypeC) {
  using KeyT = std::array<int64_t, 11>;
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

  std::stringstream ss;
  dnnl::ukernel::brgemm brg;
  brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, ldb, ldc, dnnl_dtypeA,
                              dnnl_dtypeB, dnnl_dtypeC);
  // Instruct the kernel to append the result to C tensor.
  brg.set_add_C(true);
  // Finalize the initialization.
  brg.finalize();

  bool need_packing = brg.get_B_pack_type() == pack_type::pack32;
  ss << "ldb - " << ldb << " packed_ldb - " << N << "\n";
  if (need_packing) {
    brg = dnnl::ukernel::brgemm(M, N, K_k, batch_size, lda, N, ldc, dnnl_dtypeA,
                                dnnl_dtypeB, dnnl_dtypeC);
    // Instruct the kernel to append the result to C tensor.
    brg.set_add_C(true);
    // Finalize the initialization.
    brg.finalize();

    ss << "[Packed] Args: M - " << M << ", N - " << N << ", K - " << K_k
       << ", batch - " << batch_size << ", lda - " << lda << ", ldb - " << N
       << ", ldc - " << ldc << ", dtype a - " << dtypeA << ", dtype b - "
       << dtypeB << ", dtype c - " << dtypeC << "\n";
  }
  // } else {
  //   ss << "[Unpacked] Args: M - " << M << ", N - " << N << ", K - " << K_k <<
  //   ", batch - "
  //      << batch_size << ", lda - " << lda << ", ldb - " << ldb << ", ldc - "
  //      << ldc << ", dtype a - " << dtypeA << ", dtype b - " << dtypeB
  //      << ", dtype c - " << dtypeC << "\n";
  // }
  // std::cout << ss.str();

  // Generate the executable JIT code for the objects.
  brg.generate();

  auto tf = dnnl::ukernel::transform();

  if (need_packing) {
    // Packing B tensor routine. The BRGeMM ukernel expects B passed in a
    // special VNNI format for low precision data types, e.g., bfloat16_t.
    // Note: the routine doesn't provide a `batch_size` argument in the
    // constructor as it can be either incorporated into `K` dimension, or
    // manually iterated over in a for-loop on the user side.
    dnnl::ukernel::transform pack_B(
        /* K = */ K_k, /* N = */ N,
        /* in_pack_type = */ pack_type::no_trans,
        /* in_ld = */ ldb,
        /* out_ld = */ N,
        /* in_dt = */ dnnl_dtypeB,
        /* out_dt = */ dnnl_dtypeB);

    // Pack B routine execution.
    // Note: usually should be split to process only that part of B that the
    // ukernel will execute.
    pack_B.generate();
    tf = std::move(pack_B);
  }

  auto it = savedUkernels.insert({key, {tf, brg}});
  auto ret = &it.first->second;
  return ret;
}

EXPORT void *create_transform_ukernel(int64_t K, int64_t N, int64_t in_ld,
                                      int64_t out_ld, int64_t inDtype,
                                      int64_t outDtype) {
  using K_t = std::array<int64_t, 6>;
  K_t key{K, N, in_ld, out_ld, inDtype, outDtype};
  std::stringstream ss;
  ss << "Transform args: K - " << K << ", N - " << N << ", in_ld - " << in_ld
     << ", out_ld - " << out_ld << ", in dtype - " << inDtype
     << ", out dtype - " << outDtype << "\n";
  // std::cout << ss.str();

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

void print_arr(uint16_t *ptr, int sz, std::stringstream &ss) {
  ss << "[";
  for (int i_b = 0; i_b < sz; i_b++) {
    uint16_t raw = ptr[i_b];
    uint32_t val = static_cast<uint32_t>(raw) << 16;
    ss << " " << reinterpret_cast<float &>(val);
  }
  ss << "]\n";
}

void print_arr_f(float *ptr, int sz, std::stringstream &ss) {
  ss << "[";
  for (int i_b = 0; i_b < sz; i_b++) {
    // uint16_t raw = ptr[i_b];
    // uint32_t val = static_cast<uint32_t>(raw) << 16;
    // ss << " " << reinterpret_cast<float&>(val);
    ss << " " << ptr[i_b];
  }
  ss << "]\n";
}

EXPORT void brgemm_execute(const void *handle, void *A_ptr,
                           void *original_B_ptr, void *C_ptr,
                           int64_t A_step_in_bytes, int64_t B_step_in_bytes,
                           int64_t B_block_size_in_bytes, int64_t num_batches,
                           bool skip_packing = false) {

  uint8_t *blocked_data = reinterpret_cast<uint8_t *>(original_B_ptr);
  const uint8_t *B_ptr_calc = reinterpret_cast<const uint8_t *>(original_B_ptr);

  const onednn_handle *kernel = reinterpret_cast<const onednn_handle *>(handle);

  auto pack_B = kernel->transform;
  auto brg = kernel->brg;

  bool need_packing =
      brg.get_B_pack_type() == pack_type::pack32 && !skip_packing;
  if (need_packing) {
    blocked_data = new uint8_t[B_block_size_in_bytes * num_batches];
  }

  brg.set_hw_context();

  std::vector<std::pair<memory::dim, memory::dim>> A_B_offsets(num_batches);
  std::stringstream ss;
  for (memory::dim i = 0; i < num_batches; i++) {
    const memory::dim A_offset_i = i * A_step_in_bytes;

    memory::dim B_offset_i;
    if (need_packing) {
      ss << "Requires packing!\n";
      ss << "Orig B ptr: " << (void *)B_ptr_calc
         << " packed ptr: " << (void *)blocked_data << "\n";
      ss << "Repacking: b step - " << B_step_in_bytes
         << " block size: " << B_block_size_in_bytes << "\n";
      pack_B.execute(B_ptr_calc + i * B_step_in_bytes,
                     blocked_data + i * B_block_size_in_bytes);
      B_offset_i = i * B_block_size_in_bytes;
    } else {
      B_offset_i = i * B_step_in_bytes;
    }
    ss << "pair: <" << A_offset_i << ", " << B_offset_i << ">\n";
    A_B_offsets[i] = std::make_pair(A_offset_i, B_offset_i);
  }
  // std::cout << ss.str();
  // ss.clear();
  ss << "A_ptr: " << (void *)A_ptr;
  ss << " B_ptr: " << (void *)blocked_data;
  ss << "a = \n";
  auto ptr = (uint16_t *)A_ptr;
  for (int i = 0; i < 32; i++) {
    print_arr((uint16_t *)ptr, 32, ss);
    ptr += 32;
  }
  ss << "(orig) b = \n";
  ptr = (uint16_t *)original_B_ptr;
  for (int i = 0; i < 32; i++) {
    print_arr((uint16_t *)ptr, 32, ss);
    ptr += 64;
  }
  ss << "(packed) b = \n";
  ptr = (uint16_t *)blocked_data;
  for (int i = 0; i < 32; i++) {
    print_arr((uint16_t *)ptr, 32, ss);
    ptr += 32;
  }

  // std::cout << ss.str();

  size_t scratchpad_size = brg.get_scratchpad_size();
  std::vector<uint8_t> scratchpad_sm(scratchpad_size);
  //  An execute call. `A_B` is a vector of pointers to A and packed B
  //  tensors. `acc_ptr` is a pointer to an accumulator buffer.
  brg.execute(A_ptr, blocked_data, A_B_offsets, C_ptr, scratchpad_sm.data());

  // ss.clear();
  ss << "c = \n";
  auto ptr_f = (float *)C_ptr;
  for (int i = 0; i < 32; i++) {
    print_arr_f((float *)ptr_f, 32, ss);
    ptr_f += 32;
  }

  ss << " C_ptr: " << (void *)C_ptr << "\n";
  // std::cout << ss.str();

  dnnl::ukernel::brgemm::release_hw_context();

  if (need_packing) {
    delete[] blocked_data;
  };
}

} // extern C

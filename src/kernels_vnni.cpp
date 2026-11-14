#include "simd_avx2.h"

namespace infer {
namespace {

struct VnniDot {
  static constexpr int kTokens = 4;
  static inline __m256i init_q8(int32_t sum) { return _mm256_set1_epi32(-128 * sum); }
  static inline __m256i init_u8(int32_t sum, int32_t offset) {
    return _mm256_set1_epi32(-offset * sum);
  }
  static inline void q8_weight(__m256i w, __m256i& a, __m256i& b) {
    a = _mm256_xor_si256(w, _mm256_set1_epi8(static_cast<char>(0x80)));
    b = a;
  }
  static inline __m256i q8_step(__m256i acc, __m256i a, __m256i, __m256i x) {
    return _mm256_dpbusd_avx_epi32(acc, a, x);
  }
  static inline __m256i u8_step(__m256i acc, __m256i u, __m256i x) {
    return _mm256_dpbusd_avx_epi32(acc, u, x);
  }
};

void matmul_q8(const MatmulTask& t) { matmul_quant<VnniDot, false>(t); }
void matmul_q4(const MatmulTask& t) { matmul_quant<VnniDot, true>(t); }

KernelSet make_vnni() {
  KernelSet k = *avx2_kernels();
  k.name = "vnni";
  k.matmul[static_cast<int>(WType::Q8)] = matmul_q8;
  k.matmul[static_cast<int>(WType::Q4)] = matmul_q4;
  return k;
}

}

const KernelSet* vnni_kernels() {
  static const KernelSet k = make_vnni();
  return &k;
}

}

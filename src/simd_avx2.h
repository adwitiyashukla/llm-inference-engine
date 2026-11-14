#pragma once

#include <immintrin.h>

#include <algorithm>
#include <cstring>

#include "common.h"
#include "kernels.h"

namespace infer {
namespace {

inline float hsum256(__m256 v) {
  __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
  lo = _mm_add_ss(lo, _mm_movehdup_ps(lo));
  return _mm_cvtss_f32(lo);
}

inline __m256i bcast4(const int8_t* p) {
  int32_t v;
  std::memcpy(&v, p, sizeof(v));
  return _mm256_set1_epi32(v);
}

template <class Dot, bool Q4, int N>
inline void quant_group(const uint8_t* tiles, const uint16_t* scales, int64_t nb,
                        const MatmulTask& t, int64_t tok, float* out) {
  constexpr int64_t kTile = Q4 ? kRowGroup * kQBlock / 2 : kRowGroup * kQBlock;
  const int8_t* xp[N];
  const float* dp[N];
  const int32_t* sp[N];
  __m256 f_lo[N];
  __m256 f_hi[N];
  for (int j = 0; j < N; ++j) {
    xp[j] = t.xq + (tok + j) * t.xq_stride;
    dp[j] = t.xd + (tok + j) * t.xd_stride;
    sp[j] = t.xs + (tok + j) * t.xd_stride;
    f_lo[j] = _mm256_setzero_ps();
    f_hi[j] = _mm256_setzero_ps();
  }
  const __m256i mask = _mm256_set1_epi8(0x0f);
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* tp = tiles + b * kTile;
    const __m128i* sc = reinterpret_cast<const __m128i*>(scales + b * kRowGroup);
    const __m256 s_lo = _mm256_cvtph_ps(_mm_loadu_si128(sc));
    const __m256 s_hi = _mm256_cvtph_ps(_mm_loadu_si128(sc + 1));
    __m256i a_lo[N];
    __m256i a_hi[N];
    for (int j = 0; j < N; ++j) {
      a_lo[j] = Q4 ? Dot::init_u8(sp[j][b], 8) : Dot::init_q8(sp[j][b]);
      a_hi[j] = a_lo[j];
    }
    if constexpr (Q4) {
      for (int v = 0; v < 4; ++v) {
        const __m256i p_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tp + v * 64));
        const __m256i p_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(tp + v * 64 + 32));
        const __m256i l_lo = _mm256_and_si256(p_lo, mask);
        const __m256i l_hi = _mm256_and_si256(p_hi, mask);
        const __m256i h_lo = _mm256_and_si256(_mm256_srli_epi16(p_lo, 4), mask);
        const __m256i h_hi = _mm256_and_si256(_mm256_srli_epi16(p_hi, 4), mask);
        for (int j = 0; j < N; ++j) {
          const int8_t* xb = xp[j] + b * kQBlock + v * 4;
          const __m256i x0 = bcast4(xb);
          const __m256i x1 = bcast4(xb + kQBlock / 2);
          a_lo[j] = Dot::u8_step(a_lo[j], l_lo, x0);
          a_hi[j] = Dot::u8_step(a_hi[j], l_hi, x0);
          a_lo[j] = Dot::u8_step(a_lo[j], h_lo, x1);
          a_hi[j] = Dot::u8_step(a_hi[j], h_hi, x1);
        }
      }
    } else {
      for (int k = 0; k < 8; ++k) {
        __m256i wa_lo, wb_lo, wa_hi, wb_hi;
        Dot::q8_weight(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tp + k * 64)), wa_lo,
                       wb_lo);
        Dot::q8_weight(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(tp + k * 64 + 32)),
                       wa_hi, wb_hi);
        for (int j = 0; j < N; ++j) {
          const __m256i x = bcast4(xp[j] + b * kQBlock + k * 4);
          a_lo[j] = Dot::q8_step(a_lo[j], wa_lo, wb_lo, x);
          a_hi[j] = Dot::q8_step(a_hi[j], wa_hi, wb_hi, x);
        }
      }
    }
    for (int j = 0; j < N; ++j) {
      const __m256 d = _mm256_set1_ps(dp[j][b]);
      f_lo[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a_lo[j]), _mm256_mul_ps(s_lo, d), f_lo[j]);
      f_hi[j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a_hi[j]), _mm256_mul_ps(s_hi, d), f_hi[j]);
    }
  }
  for (int j = 0; j < N; ++j) {
    _mm256_storeu_ps(out + j * kRowGroup, f_lo[j]);
    _mm256_storeu_ps(out + j * kRowGroup + 8, f_hi[j]);
  }
}

inline void store_group(const MatmulTask& t, const float* out, int64_t tok, int count, int64_t g) {
  const int64_t r0 = g * kRowGroup;
  const int64_t lo = std::max(r0, t.row_begin);
  const int64_t hi = std::min(r0 + kRowGroup, t.row_end);
  for (int j = 0; j < count; ++j) {
    float* y = t.y + (tok + j) * t.y_stride;
    for (int64_t r = lo; r < hi; ++r) y[r] = out[j * kRowGroup + (r - r0)];
  }
}

template <class Dot, bool Q4>
void matmul_quant(const MatmulTask& t) {
  constexpr int kTokens = Dot::kTokens;
  const WeightView& w = *t.w;
  const int64_t nb = w.cols / kQBlock;
  const int64_t tile = Q4 ? kRowGroup * kQBlock / 2 : kRowGroup * kQBlock;
  const uint8_t* base = static_cast<const uint8_t*>(w.data);
  alignas(32) float out[kTokens * kRowGroup];
  for (int64_t g = t.row_begin / kRowGroup; g * kRowGroup < t.row_end; ++g) {
    const uint8_t* tiles = base + g * nb * tile;
    const uint16_t* scales = w.scales + g * nb * kRowGroup;
    int64_t tok = 0;
    for (; tok + kTokens <= t.tokens; tok += kTokens) {
      quant_group<Dot, Q4, kTokens>(tiles, scales, nb, t, tok, out);
      store_group(t, out, tok, kTokens, g);
    }
    const int rest = static_cast<int>(t.tokens - tok);
    if (rest == 0) continue;
    if (rest == 1) quant_group<Dot, Q4, 1>(tiles, scales, nb, t, tok, out);
    if (rest == 2) quant_group<Dot, Q4, 2>(tiles, scales, nb, t, tok, out);
    if constexpr (kTokens > 3) {
      if (rest == 3) quant_group<Dot, Q4, 3>(tiles, scales, nb, t, tok, out);
    }
    store_group(t, out, tok, rest, g);
  }
}

}
}

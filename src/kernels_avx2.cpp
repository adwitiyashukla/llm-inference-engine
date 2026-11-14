#include <algorithm>
#include <cmath>

#include "simd_avx2.h"

namespace infer {
namespace {

struct MaddDot {
  static constexpr int kTokens = 3;
  static inline __m256i init_q8(int32_t) { return _mm256_setzero_si256(); }
  static inline __m256i init_u8(int32_t sum, int32_t offset) {
    return _mm256_set1_epi32(-offset * sum);
  }
  static inline void q8_weight(__m256i w, __m256i& a, __m256i& b) {
    a = _mm256_sign_epi8(w, w);
    b = w;
  }
  static inline __m256i q8_step(__m256i acc, __m256i a, __m256i b, __m256i x) {
    __m256i p = _mm256_maddubs_epi16(a, _mm256_sign_epi8(x, b));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
  }
  static inline __m256i u8_step(__m256i acc, __m256i u, __m256i x) {
    __m256i p = _mm256_maddubs_epi16(u, x);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
  }
};

inline __m256 load8_f32(const float* p) { return _mm256_loadu_ps(p); }

inline __m256 load8_bf16(const uint16_t* p) {
  __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  return _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
}

inline __m256 load8_f16(const uint16_t* p) {
  return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

template <class T, class Load, class Scalar, int N>
inline void float_tile(const T* wr, int64_t cols, const MatmulTask& t, int64_t tok, int64_t r,
                       Load load, Scalar scalar) {
  const float* xp[N];
  __m256 a0[N];
  __m256 a1[N];
  for (int j = 0; j < N; ++j) {
    xp[j] = t.x + (tok + j) * t.x_stride;
    a0[j] = _mm256_setzero_ps();
    a1[j] = _mm256_setzero_ps();
  }
  int64_t c = 0;
  for (; c + 16 <= cols; c += 16) {
    __m256 w0 = load(wr + c);
    __m256 w1 = load(wr + c + 8);
    for (int j = 0; j < N; ++j) {
      a0[j] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp[j] + c), a0[j]);
      a1[j] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xp[j] + c + 8), a1[j]);
    }
  }
  if (c + 8 <= cols) {
    __m256 w0 = load(wr + c);
    for (int j = 0; j < N; ++j) a0[j] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xp[j] + c), a0[j]);
    c += 8;
  }
  for (int j = 0; j < N; ++j) {
    float sum = hsum256(_mm256_add_ps(a0[j], a1[j]));
    for (int64_t k = c; k < cols; ++k) sum += scalar(wr[k]) * xp[j][k];
    t.y[(tok + j) * t.y_stride + r] = sum;
  }
}

template <class T, class Load, class Scalar>
void matmul_float(const MatmulTask& t, Load load, Scalar scalar) {
  constexpr int64_t kRows = 8;
  const WeightView& w = *t.w;
  const T* base = static_cast<const T*>(w.data);
  for (int64_t r0 = t.row_begin; r0 < t.row_end; r0 += kRows) {
    const int64_t r1 = std::min(r0 + kRows, t.row_end);
    int64_t tok = 0;
    for (; tok + 4 <= t.tokens; tok += 4) {
      for (int64_t r = r0; r < r1; ++r) {
        float_tile<T, Load, Scalar, 4>(base + r * w.cols, w.cols, t, tok, r, load, scalar);
      }
    }
    for (; tok < t.tokens; ++tok) {
      for (int64_t r = r0; r < r1; ++r) {
        float_tile<T, Load, Scalar, 1>(base + r * w.cols, w.cols, t, tok, r, load, scalar);
      }
    }
  }
}

void matmul_f32(const MatmulTask& t) {
  matmul_float<float>(t, load8_f32, [](float v) { return v; });
}

void matmul_f16(const MatmulTask& t) {
  matmul_float<uint16_t>(t, load8_f16, [](uint16_t v) { return f16_to_f32(v); });
}

void matmul_bf16(const MatmulTask& t) {
  matmul_float<uint16_t>(t, load8_bf16, [](uint16_t v) { return bf16_to_f32(v); });
}

void matmul_q8(const MatmulTask& t) { matmul_quant<MaddDot, false>(t); }
void matmul_q4(const MatmulTask& t) { matmul_quant<MaddDot, true>(t); }

inline int32_t hsum_epi32(__m256i v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

void quantize_row(const float* x, int64_t n, int8_t* q, float* d, int32_t* s) {
  const __m256 sign = _mm256_set1_ps(-0.0f);
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x + b * kQBlock;
    __m256 v0 = _mm256_loadu_ps(xb);
    __m256 v1 = _mm256_loadu_ps(xb + 8);
    __m256 v2 = _mm256_loadu_ps(xb + 16);
    __m256 v3 = _mm256_loadu_ps(xb + 24);
    __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                             _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_movehdup_ps(m4));
    float amax = _mm_cvtss_f32(m4);
    float scale = amax / 127.0f;
    float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    __m256 iv = _mm256_set1_ps(inv);
    const int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
    __m256i i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, iv), mode));
    __m256i i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, iv), mode));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, iv), mode));
    __m256i i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, iv), mode));
    __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
    packed = _mm256_permutevar8x32_epi32(packed, order);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(q + b * kQBlock), packed);
    d[b] = scale;
    s[b] = hsum_epi32(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
  }
}

template <class Load>
inline float dot_impl(const float* a, const uint16_t* bh, const float* bf, int64_t n, Load load) {
  __m256 s0 = _mm256_setzero_ps();
  __m256 s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps();
  __m256 s3 = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 32 <= n; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load(bh, bf, i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), load(bh, bf, i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), load(bh, bf, i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), load(bh, bf, i + 24), s3);
  }
  for (; i + 8 <= n; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), load(bh, bf, i), s0);
  float sum = hsum256(_mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3)));
  for (; i < n; ++i) sum += a[i] * (bf != nullptr ? bf[i] : f16_to_f32(bh[i]));
  return sum;
}

float dot(const float* a, const float* b, int64_t n) {
  return dot_impl(a, nullptr, b, n, [](const uint16_t*, const float* p, int64_t i) {
    return _mm256_loadu_ps(p + i);
  });
}

float dot_f16(const float* a, const uint16_t* b, int64_t n) {
  return dot_impl(a, b, nullptr, n,
                  [](const uint16_t* p, const float*, int64_t i) { return load8_f16(p + i); });
}

void axpy(float* y, float a, const float* x, int64_t n) {
  __m256 av = _mm256_set1_ps(a);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] += a * x[i];
}

void axpy_f16(float* y, float a, const uint16_t* x, int64_t n) {
  __m256 av = _mm256_set1_ps(a);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, load8_f16(x + i), _mm256_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] += a * f16_to_f32(x[i]);
}

void to_f16(const float* x, uint16_t* y, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(y + i), h);
  }
  for (; i < n; ++i) y[i] = f32_to_f16(x[i]);
}

inline __m256 exp256(__m256 x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
  __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
  r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
  __m256 p = _mm256_set1_ps(1.9875691500e-4f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073e-3f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894e-2f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459e-1f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201e-1f));
  p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.0f)));
  __m256i e =
      _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

float exp_shifted(float* x, int64_t n, float shift) {
  __m256 sv = _mm256_set1_ps(shift);
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(x + i), sv));
    _mm256_storeu_ps(x + i, e);
    acc = _mm256_add_ps(acc, e);
  }
  float sum = hsum256(acc);
  for (; i < n; ++i) {
    x[i] = std::exp(x[i] - shift);
    sum += x[i];
  }
  return sum;
}

void silu(float* x, int64_t n) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 neg = _mm256_set1_ps(-0.0f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(x + i);
    _mm256_storeu_ps(x + i, _mm256_div_ps(g, _mm256_add_ps(one, exp256(_mm256_xor_ps(g, neg)))));
  }
  for (; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 neg = _mm256_set1_ps(-0.0f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 s = _mm256_div_ps(g, _mm256_add_ps(one, exp256(_mm256_xor_ps(g, neg))));
    _mm256_storeu_ps(out + i, _mm256_mul_ps(s, _mm256_loadu_ps(up + i)));
  }
  for (; i < n; ++i) out[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

void sigmoid_mul(float* x, const float* gate, int64_t n) {
  const __m256 one = _mm256_set1_ps(1.0f);
  const __m256 neg = _mm256_set1_ps(-0.0f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 g = _mm256_loadu_ps(gate + i);
    __m256 s = _mm256_div_ps(one, _mm256_add_ps(one, exp256(_mm256_xor_ps(g, neg))));
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), s));
  }
  for (; i < n; ++i) x[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
}

inline void delta_block(float* state, int64_t dv, int64_t j, const float* q, const float* k,
                        const float* v, float decay, float beta, float* out, int64_t dk) {
  const __m256 dec = _mm256_set1_ps(decay);
  const __m256 bt = _mm256_set1_ps(beta);
  __m256 m[4];
  for (auto& x : m) x = _mm256_setzero_ps();
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv + j;
    __m256 ki = _mm256_set1_ps(k[i]);
    for (int u = 0; u < 4; ++u) {
      __m256 s = _mm256_mul_ps(_mm256_loadu_ps(row + 8 * u), dec);
      _mm256_storeu_ps(row + 8 * u, s);
      m[u] = _mm256_fmadd_ps(s, ki, m[u]);
    }
  }
  __m256 d[4];
  __m256 o[4];
  for (int u = 0; u < 4; ++u) {
    d[u] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + j + 8 * u), m[u]), bt);
    o[u] = _mm256_setzero_ps();
  }
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv + j;
    __m256 ki = _mm256_set1_ps(k[i]);
    __m256 qi = _mm256_set1_ps(q[i]);
    for (int u = 0; u < 4; ++u) {
      __m256 s = _mm256_fmadd_ps(ki, d[u], _mm256_loadu_ps(row + 8 * u));
      _mm256_storeu_ps(row + 8 * u, s);
      o[u] = _mm256_fmadd_ps(s, qi, o[u]);
    }
  }
  for (int u = 0; u < 4; ++u) _mm256_storeu_ps(out + j + 8 * u, o[u]);
}

void deltanet_step(float* state, const float* q, const float* k, const float* v, float decay,
                   float beta, float* out, float* scratch, int64_t dk, int64_t dv) {
  if (dv % 32 != 0) {
    scalar_kernels()->deltanet_step(state, q, k, v, decay, beta, out, scratch, dk, dv);
    return;
  }
  for (int64_t j = 0; j < dv; j += 32) delta_block(state, dv, j, q, k, v, decay, beta, out, dk);
}

void deltanet_seq(float* state, const DeltaSeq& s, float* scratch, int64_t dk, int64_t dv) {
  constexpr int kRegs = 2;
  constexpr int64_t kWidth = 16;
  if (dv % kWidth != 0 || s.tokens == 0) {
    scalar_kernels()->deltanet_seq(state, s, scratch, dk, dv);
    return;
  }
  for (int64_t j = 0; j < dv; j += kWidth) {
    __m256 m[kRegs];
    __m256 d[kRegs];
    __m256 o[kRegs];
    {
      const __m256 dec = _mm256_set1_ps(s.gates[1]);
      for (auto& x : m) x = _mm256_setzero_ps();
      for (int64_t i = 0; i < dk; ++i) {
        float* row = state + i * dv + j;
        const __m256 ki = _mm256_set1_ps(s.k[i]);
        for (int u = 0; u < kRegs; ++u) {
          __m256 v = _mm256_mul_ps(_mm256_loadu_ps(row + 8 * u), dec);
          _mm256_storeu_ps(row + 8 * u, v);
          m[u] = _mm256_fmadd_ps(v, ki, m[u]);
        }
      }
    }
    for (int64_t t = 0; t < s.tokens; ++t) {
      const float* q = s.q + t * s.qkv_stride;
      const float* k = s.k + t * s.qkv_stride;
      const float* v = s.v + t * s.qkv_stride;
      const __m256 bt = _mm256_set1_ps(s.gates[t * s.gate_stride]);
      for (int u = 0; u < kRegs; ++u) {
        d[u] = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(v + j + 8 * u), m[u]), bt);
        o[u] = _mm256_setzero_ps();
        m[u] = _mm256_setzero_ps();
      }
      const bool last = t + 1 == s.tokens;
      const float* kn = last ? k : s.k + (t + 1) * s.qkv_stride;
      const __m256 dec = _mm256_set1_ps(last ? 1.0f : s.gates[(t + 1) * s.gate_stride + 1]);
      for (int64_t i = 0; i < dk; ++i) {
        float* row = state + i * dv + j;
        const __m256 ki = _mm256_set1_ps(k[i]);
        const __m256 qi = _mm256_set1_ps(q[i]);
        const __m256 kn_i = _mm256_set1_ps(kn[i]);
        for (int u = 0; u < kRegs; ++u) {
          __m256 x = _mm256_fmadd_ps(ki, d[u], _mm256_loadu_ps(row + 8 * u));
          o[u] = _mm256_fmadd_ps(x, qi, o[u]);
          if (!last) {
            x = _mm256_mul_ps(x, dec);
            m[u] = _mm256_fmadd_ps(x, kn_i, m[u]);
          }
          _mm256_storeu_ps(row + 8 * u, x);
        }
      }
      float* out = s.out + t * s.out_stride;
      for (int u = 0; u < kRegs; ++u) _mm256_storeu_ps(out + j + 8 * u, o[u]);
    }
  }
}

const KernelSet kAvx2 = {
    "avx2",        {matmul_f32, matmul_f16, matmul_bf16, matmul_q8, matmul_q4},
    quantize_row,  dot,
    dot_f16,       axpy,
    axpy_f16,      to_f16,
    exp_shifted,   silu,
    silu_mul,      sigmoid_mul,
    deltanet_step, deltanet_seq,
};

}

const KernelSet* avx2_kernels() { return &kAvx2; }

}

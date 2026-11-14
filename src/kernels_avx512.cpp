#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "kernels.h"

namespace infer {
namespace {

inline __m512i bcast4(const int8_t* p) {
  int32_t v;
  std::memcpy(&v, p, sizeof(v));
  return _mm512_set1_epi32(v);
}

template <bool Q4, int N>
inline void quant_group(const uint8_t* tiles, const uint16_t* scales, int64_t nb,
                        const MatmulTask& t, int64_t tok, float* out) {
  constexpr int64_t kTile = Q4 ? kRowGroup * kQBlock / 2 : kRowGroup * kQBlock;
  const int8_t* xp[N];
  const float* dp[N];
  const int32_t* sp[N];
  __m512 f[N];
  for (int j = 0; j < N; ++j) {
    xp[j] = t.xq + (tok + j) * t.xq_stride;
    dp[j] = t.xd + (tok + j) * t.xd_stride;
    sp[j] = t.xs + (tok + j) * t.xd_stride;
    f[j] = _mm512_setzero_ps();
  }
  const __m512i mask = _mm512_set1_epi8(0x0f);
  const __m512i flip = _mm512_set1_epi8(static_cast<char>(0x80));
  for (int64_t b = 0; b < nb; ++b) {
    const uint8_t* tp = tiles + b * kTile;
    const __m512 sc = _mm512_cvtph_ps(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales + b * kRowGroup)));
    __m512i a[N];
    for (int j = 0; j < N; ++j) a[j] = _mm512_set1_epi32(-(Q4 ? 8 : 128) * sp[j][b]);
    if constexpr (Q4) {
      for (int v = 0; v < 4; ++v) {
        const __m512i p = _mm512_loadu_si512(tp + v * 64);
        const __m512i lo = _mm512_and_si512(p, mask);
        const __m512i hi = _mm512_and_si512(_mm512_srli_epi16(p, 4), mask);
        for (int j = 0; j < N; ++j) {
          const int8_t* xb = xp[j] + b * kQBlock + v * 4;
          a[j] = _mm512_dpbusd_epi32(a[j], lo, bcast4(xb));
          a[j] = _mm512_dpbusd_epi32(a[j], hi, bcast4(xb + kQBlock / 2));
        }
      }
    } else {
      for (int k = 0; k < 8; ++k) {
        const __m512i wu = _mm512_xor_si512(_mm512_loadu_si512(tp + k * 64), flip);
        for (int j = 0; j < N; ++j) {
          a[j] = _mm512_dpbusd_epi32(a[j], wu, bcast4(xp[j] + b * kQBlock + k * 4));
        }
      }
    }
    for (int j = 0; j < N; ++j) {
      f[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[j]), _mm512_mul_ps(sc, _mm512_set1_ps(dp[j][b])),
                             f[j]);
    }
  }
  for (int j = 0; j < N; ++j) _mm512_storeu_ps(out + j * kRowGroup, f[j]);
}

template <bool Q4>
void matmul_quant(const MatmulTask& t) {
  constexpr int kTokens = 6;
  const WeightView& w = *t.w;
  const int64_t nb = w.cols / kQBlock;
  const int64_t tile = Q4 ? kRowGroup * kQBlock / 2 : kRowGroup * kQBlock;
  const uint8_t* base = static_cast<const uint8_t*>(w.data);
  alignas(64) float out[kTokens * kRowGroup];
  auto store = [&](int64_t tok, int count, int64_t g) {
    const int64_t r0 = g * kRowGroup;
    const int64_t lo = std::max(r0, t.row_begin);
    const int64_t hi = std::min(r0 + kRowGroup, t.row_end);
    for (int j = 0; j < count; ++j) {
      float* y = t.y + (tok + j) * t.y_stride;
      for (int64_t r = lo; r < hi; ++r) y[r] = out[j * kRowGroup + (r - r0)];
    }
  };
  for (int64_t g = t.row_begin / kRowGroup; g * kRowGroup < t.row_end; ++g) {
    const uint8_t* tiles = base + g * nb * tile;
    const uint16_t* scales = w.scales + g * nb * kRowGroup;
    int64_t tok = 0;
    for (; tok + kTokens <= t.tokens; tok += kTokens) {
      quant_group<Q4, kTokens>(tiles, scales, nb, t, tok, out);
      store(tok, kTokens, g);
    }
    const int rest = static_cast<int>(t.tokens - tok);
    if (rest == 0) continue;
    if (rest == 1) quant_group<Q4, 1>(tiles, scales, nb, t, tok, out);
    if (rest == 2) quant_group<Q4, 2>(tiles, scales, nb, t, tok, out);
    if (rest == 3) quant_group<Q4, 3>(tiles, scales, nb, t, tok, out);
    if (rest == 4) quant_group<Q4, 4>(tiles, scales, nb, t, tok, out);
    if (rest == 5) quant_group<Q4, 5>(tiles, scales, nb, t, tok, out);
    store(tok, rest, g);
  }
}

void matmul_q8(const MatmulTask& t) { matmul_quant<false>(t); }
void matmul_q4(const MatmulTask& t) { matmul_quant<true>(t); }

inline __m512 load16_f32(const float* p) { return _mm512_loadu_ps(p); }

inline __m512 load16_bf16(const uint16_t* p) {
  __m512i v = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
  return _mm512_castsi512_ps(_mm512_slli_epi32(v, 16));
}

inline __m512 load16_f16(const uint16_t* p) {
  return _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

template <class T, class Load, class Scalar, int N>
inline void float_tile(const T* wr, int64_t cols, const MatmulTask& t, int64_t tok, int64_t r,
                       Load load, Scalar scalar) {
  const float* xp[N];
  __m512 a0[N];
  __m512 a1[N];
  for (int j = 0; j < N; ++j) {
    xp[j] = t.x + (tok + j) * t.x_stride;
    a0[j] = _mm512_setzero_ps();
    a1[j] = _mm512_setzero_ps();
  }
  int64_t c = 0;
  for (; c + 32 <= cols; c += 32) {
    __m512 w0 = load(wr + c);
    __m512 w1 = load(wr + c + 16);
    for (int j = 0; j < N; ++j) {
      a0[j] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xp[j] + c), a0[j]);
      a1[j] = _mm512_fmadd_ps(w1, _mm512_loadu_ps(xp[j] + c + 16), a1[j]);
    }
  }
  if (c + 16 <= cols) {
    __m512 w0 = load(wr + c);
    for (int j = 0; j < N; ++j) a0[j] = _mm512_fmadd_ps(w0, _mm512_loadu_ps(xp[j] + c), a0[j]);
    c += 16;
  }
  for (int j = 0; j < N; ++j) {
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(a0[j], a1[j]));
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
  matmul_float<float>(t, load16_f32, [](float v) { return v; });
}

void matmul_f16(const MatmulTask& t) {
  matmul_float<uint16_t>(t, load16_f16, [](uint16_t v) { return f16_to_f32(v); });
}

void matmul_bf16(const MatmulTask& t) {
  matmul_float<uint16_t>(t, load16_bf16, [](uint16_t v) { return bf16_to_f32(v); });
}

void quantize_row(const float* x, int64_t n, int8_t* q, float* d, int32_t* s) {
  const int mode = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x + b * kQBlock;
    __m512 v0 = _mm512_loadu_ps(xb);
    __m512 v1 = _mm512_loadu_ps(xb + 16);
    float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
    float scale = amax / 127.0f;
    float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    __m512 iv = _mm512_set1_ps(inv);
    __m512i i0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v0, iv), mode));
    __m512i i1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v1, iv), mode));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + b * kQBlock), _mm512_cvtsepi32_epi8(i0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + b * kQBlock + 16), _mm512_cvtsepi32_epi8(i1));
    d[b] = scale;
    s[b] = _mm512_reduce_add_epi32(_mm512_add_epi32(i0, i1));
  }
}

template <class Load>
inline float dot_impl(const float* a, int64_t n, Load load) {
  __m512 s0 = _mm512_setzero_ps();
  __m512 s1 = _mm512_setzero_ps();
  __m512 s2 = _mm512_setzero_ps();
  __m512 s3 = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 64 <= n; i += 64) {
    s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), load(i), s0);
    s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), load(i + 16), s1);
    s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), load(i + 32), s2);
    s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), load(i + 48), s3);
  }
  for (; i + 16 <= n; i += 16) s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), load(i), s0);
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)));
}

float dot(const float* a, const float* b, int64_t n) {
  float sum = dot_impl(a, n, [b](int64_t i) { return _mm512_loadu_ps(b + i); });
  for (int64_t i = n - n % 16; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

float dot_f16(const float* a, const uint16_t* b, int64_t n) {
  float sum = dot_impl(a, n, [b](int64_t i) { return load16_f16(b + i); });
  for (int64_t i = n - n % 16; i < n; ++i) sum += a[i] * f16_to_f32(b[i]);
  return sum;
}

void axpy(float* y, float a, const float* x, int64_t n) {
  __m512 av = _mm512_set1_ps(a);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, _mm512_loadu_ps(x + i), _mm512_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] += a * x[i];
}

void axpy_f16(float* y, float a, const uint16_t* x, int64_t n) {
  __m512 av = _mm512_set1_ps(a);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, load16_f16(x + i), _mm512_loadu_ps(y + i)));
  }
  for (; i < n; ++i) y[i] += a * f16_to_f32(x[i]);
}

void to_f16(const float* x, uint16_t* y, int64_t n) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m256i h = _mm512_cvtps_ph(_mm512_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(y + i), h);
  }
  for (; i < n; ++i) y[i] = f32_to_f16(x[i]);
}

inline __m512 exp512(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(88.0f));
  __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
  __m512 p = _mm512_set1_ps(1.9875691500e-4f);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
  p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
  __m512i e =
      _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(p, _mm512_castsi512_ps(e));
}

float exp_shifted(float* x, int64_t n, float shift) {
  __m512 sv = _mm512_set1_ps(shift);
  __m512 acc = _mm512_setzero_ps();
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 e = exp512(_mm512_sub_ps(_mm512_loadu_ps(x + i), sv));
    _mm512_storeu_ps(x + i, e);
    acc = _mm512_add_ps(acc, e);
  }
  float sum = _mm512_reduce_add_ps(acc);
  for (; i < n; ++i) {
    x[i] = std::exp(x[i] - shift);
    sum += x[i];
  }
  return sum;
}

void silu(float* x, int64_t n) {
  const __m512 one = _mm512_set1_ps(1.0f);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 g = _mm512_loadu_ps(x + i);
    _mm512_storeu_ps(
        x + i, _mm512_div_ps(g, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), g)))));
  }
  for (; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
  const __m512 one = _mm512_set1_ps(1.0f);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 g = _mm512_loadu_ps(gate + i);
    __m512 s = _mm512_div_ps(g, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), g))));
    _mm512_storeu_ps(out + i, _mm512_mul_ps(s, _mm512_loadu_ps(up + i)));
  }
  for (; i < n; ++i) out[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

void sigmoid_mul(float* x, const float* gate, int64_t n) {
  const __m512 one = _mm512_set1_ps(1.0f);
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 g = _mm512_loadu_ps(gate + i);
    __m512 s =
        _mm512_div_ps(one, _mm512_add_ps(one, exp512(_mm512_sub_ps(_mm512_setzero_ps(), g))));
    _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_loadu_ps(x + i), s));
  }
  for (; i < n; ++i) x[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
}

inline void delta_block(float* state, int64_t dv, int64_t j, const float* q, const float* k,
                        const float* v, float decay, float beta, float* out, int64_t dk) {
  const __m512 dec = _mm512_set1_ps(decay);
  const __m512 bt = _mm512_set1_ps(beta);
  __m512 m[4];
  for (auto& x : m) x = _mm512_setzero_ps();
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv + j;
    __m512 ki = _mm512_set1_ps(k[i]);
    for (int u = 0; u < 4; ++u) {
      __m512 s = _mm512_mul_ps(_mm512_loadu_ps(row + 16 * u), dec);
      _mm512_storeu_ps(row + 16 * u, s);
      m[u] = _mm512_fmadd_ps(s, ki, m[u]);
    }
  }
  __m512 d[4];
  __m512 o[4];
  for (int u = 0; u < 4; ++u) {
    d[u] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v + j + 16 * u), m[u]), bt);
    o[u] = _mm512_setzero_ps();
  }
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv + j;
    __m512 ki = _mm512_set1_ps(k[i]);
    __m512 qi = _mm512_set1_ps(q[i]);
    for (int u = 0; u < 4; ++u) {
      __m512 s = _mm512_fmadd_ps(ki, d[u], _mm512_loadu_ps(row + 16 * u));
      _mm512_storeu_ps(row + 16 * u, s);
      o[u] = _mm512_fmadd_ps(s, qi, o[u]);
    }
  }
  for (int u = 0; u < 4; ++u) _mm512_storeu_ps(out + j + 16 * u, o[u]);
}

void deltanet_step(float* state, const float* q, const float* k, const float* v, float decay,
                   float beta, float* out, float* scratch, int64_t dk, int64_t dv) {
  if (dv % 64 != 0) {
    scalar_kernels()->deltanet_step(state, q, k, v, decay, beta, out, scratch, dk, dv);
    return;
  }
  for (int64_t j = 0; j < dv; j += 64) delta_block(state, dv, j, q, k, v, decay, beta, out, dk);
}

void deltanet_seq(float* state, const DeltaSeq& s, float* scratch, int64_t dk, int64_t dv) {
  constexpr int kRegs = 4;
  constexpr int64_t kWidth = 64;
  if (dv % kWidth != 0 || s.tokens == 0) {
    scalar_kernels()->deltanet_seq(state, s, scratch, dk, dv);
    return;
  }
  for (int64_t j = 0; j < dv; j += kWidth) {
    __m512 m[kRegs];
    __m512 d[kRegs];
    __m512 o[kRegs];
    {
      const __m512 dec = _mm512_set1_ps(s.gates[1]);
      for (auto& x : m) x = _mm512_setzero_ps();
      for (int64_t i = 0; i < dk; ++i) {
        float* row = state + i * dv + j;
        const __m512 ki = _mm512_set1_ps(s.k[i]);
        for (int u = 0; u < kRegs; ++u) {
          __m512 v = _mm512_mul_ps(_mm512_loadu_ps(row + 16 * u), dec);
          _mm512_storeu_ps(row + 16 * u, v);
          m[u] = _mm512_fmadd_ps(v, ki, m[u]);
        }
      }
    }
    for (int64_t t = 0; t < s.tokens; ++t) {
      const float* q = s.q + t * s.qkv_stride;
      const float* k = s.k + t * s.qkv_stride;
      const float* v = s.v + t * s.qkv_stride;
      const __m512 bt = _mm512_set1_ps(s.gates[t * s.gate_stride]);
      for (int u = 0; u < kRegs; ++u) {
        d[u] = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v + j + 16 * u), m[u]), bt);
        o[u] = _mm512_setzero_ps();
        m[u] = _mm512_setzero_ps();
      }
      const bool last = t + 1 == s.tokens;
      const float* kn = last ? k : s.k + (t + 1) * s.qkv_stride;
      const __m512 dec = _mm512_set1_ps(last ? 1.0f : s.gates[(t + 1) * s.gate_stride + 1]);
      for (int64_t i = 0; i < dk; ++i) {
        float* row = state + i * dv + j;
        const __m512 ki = _mm512_set1_ps(k[i]);
        const __m512 qi = _mm512_set1_ps(q[i]);
        const __m512 kn_i = _mm512_set1_ps(kn[i]);
        for (int u = 0; u < kRegs; ++u) {
          __m512 x = _mm512_fmadd_ps(ki, d[u], _mm512_loadu_ps(row + 16 * u));
          o[u] = _mm512_fmadd_ps(x, qi, o[u]);
          if (!last) {
            x = _mm512_mul_ps(x, dec);
            m[u] = _mm512_fmadd_ps(x, kn_i, m[u]);
          }
          _mm512_storeu_ps(row + 16 * u, x);
        }
      }
      float* out = s.out + t * s.out_stride;
      for (int u = 0; u < kRegs; ++u) _mm512_storeu_ps(out + j + 16 * u, o[u]);
    }
  }
}

const KernelSet kAvx512 = {
    "avx512",      {matmul_f32, matmul_f16, matmul_bf16, matmul_q8, matmul_q4},
    quantize_row,  dot,
    dot_f16,       axpy,
    axpy_f16,      to_f16,
    exp_shifted,   silu,
    silu_mul,      sigmoid_mul,
    deltanet_step, deltanet_seq,
};

}

const KernelSet* avx512_kernels() { return &kAvx512; }

}

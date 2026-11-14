#include <algorithm>
#include <cmath>

#include "common.h"
#include "kernels.h"

namespace infer {
namespace {

template <bool Q4>
void matmul_quant(const MatmulTask& t) {
  const WeightView& w = *t.w;
  const int64_t nb = w.cols / kQBlock;
  const int64_t tile_bytes = Q4 ? kRowGroup * kQBlock / 2 : kRowGroup * kQBlock;
  const int64_t vec = kRowGroup * 4;
  const uint8_t* base = static_cast<const uint8_t*>(w.data);
  for (int64_t r = t.row_begin; r < t.row_end; ++r) {
    const int64_t g = r / kRowGroup;
    const int64_t lane = (r % kRowGroup) * 4;
    for (int64_t tok = 0; tok < t.tokens; ++tok) {
      const int8_t* xr = t.xq + tok * t.xq_stride;
      const float* xd = t.xd + tok * t.xd_stride;
      float acc = 0.0f;
      for (int64_t b = 0; b < nb; ++b) {
        const uint8_t* tile = base + (g * nb + b) * tile_bytes;
        const int8_t* xb = xr + b * kQBlock;
        int32_t sum = 0;
        if (Q4) {
          for (int64_t c = 0; c < kQBlock / 2; ++c) {
            uint8_t v = tile[(c / 4) * vec + lane + c % 4];
            sum += (static_cast<int32_t>(v & 0x0f) - 8) * xb[c];
            sum += (static_cast<int32_t>(v >> 4) - 8) * xb[c + kQBlock / 2];
          }
        } else {
          for (int64_t c = 0; c < kQBlock; ++c) {
            sum += static_cast<int32_t>(static_cast<int8_t>(tile[(c / 4) * vec + lane + c % 4])) *
                   xb[c];
          }
        }
        float scale = f16_to_f32(w.scales[(g * nb + b) * kRowGroup + r % kRowGroup]) * xd[b];
        acc = std::fma(static_cast<float>(sum), scale, acc);
      }
      t.y[tok * t.y_stride + r] = acc;
    }
  }
}

void matmul_q8(const MatmulTask& t) { matmul_quant<false>(t); }
void matmul_q4(const MatmulTask& t) { matmul_quant<true>(t); }

template <class Load>
void matmul_float(const MatmulTask& t, Load load) {
  const WeightView& w = *t.w;
  for (int64_t r = t.row_begin; r < t.row_end; ++r) {
    for (int64_t tok = 0; tok < t.tokens; ++tok) {
      const float* xr = t.x + tok * t.x_stride;
      float acc = 0.0f;
      for (int64_t c = 0; c < w.cols; ++c) acc += load(r * w.cols + c) * xr[c];
      t.y[tok * t.y_stride + r] = acc;
    }
  }
}

void matmul_f32(const MatmulTask& t) {
  const float* p = static_cast<const float*>(t.w->data);
  matmul_float(t, [p](int64_t i) { return p[i]; });
}

void matmul_f16(const MatmulTask& t) {
  const uint16_t* p = static_cast<const uint16_t*>(t.w->data);
  matmul_float(t, [p](int64_t i) { return f16_to_f32(p[i]); });
}

void matmul_bf16(const MatmulTask& t) {
  const uint16_t* p = static_cast<const uint16_t*>(t.w->data);
  matmul_float(t, [p](int64_t i) { return bf16_to_f32(p[i]); });
}

void quantize_row(const float* x, int64_t n, int8_t* q, float* d, int32_t* s) {
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x + b * kQBlock;
    float amax = 0.0f;
    for (int64_t i = 0; i < kQBlock; ++i) amax = std::max(amax, std::fabs(xb[i]));
    float scale = amax / 127.0f;
    float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
    int32_t sum = 0;
    for (int64_t i = 0; i < kQBlock; ++i) {
      int v = static_cast<int>(std::nearbyint(xb[i] * inv));
      q[b * kQBlock + i] = static_cast<int8_t>(v);
      sum += v;
    }
    d[b] = scale;
    s[b] = sum;
  }
}

float dot(const float* a, const float* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; ++i) acc += a[i] * b[i];
  return acc;
}

float dot_f16(const float* a, const uint16_t* b, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = 0; i < n; ++i) acc += a[i] * f16_to_f32(b[i]);
  return acc;
}

void axpy(float* y, float a, const float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += a * x[i];
}

void axpy_f16(float* y, float a, const uint16_t* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] += a * f16_to_f32(x[i]);
}

void to_f16(const float* x, uint16_t* y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = f32_to_f16(x[i]);
}

float exp_shifted(float* x, int64_t n, float shift) {
  float sum = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - shift);
    sum += x[i];
  }
  return sum;
}

void silu(float* x, int64_t n) {
  for (int64_t i = 0; i < n; ++i) x[i] = x[i] / (1.0f + std::exp(-x[i]));
}

void silu_mul(const float* gate, const float* up, float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
}

void sigmoid_mul(float* x, const float* gate, int64_t n) {
  for (int64_t i = 0; i < n; ++i) x[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
}

void deltanet_step(float* state, const float* q, const float* k, const float* v, float decay,
                   float beta, float* out, float* scratch, int64_t dk, int64_t dv) {
  float* mem = scratch;
  for (int64_t j = 0; j < dv; ++j) mem[j] = 0.0f;
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv;
    float ki = k[i];
    for (int64_t j = 0; j < dv; ++j) {
      row[j] *= decay;
      mem[j] += row[j] * ki;
    }
  }
  for (int64_t j = 0; j < dv; ++j) {
    mem[j] = (v[j] - mem[j]) * beta;
    out[j] = 0.0f;
  }
  for (int64_t i = 0; i < dk; ++i) {
    float* row = state + i * dv;
    float ki = k[i];
    float qi = q[i];
    for (int64_t j = 0; j < dv; ++j) {
      row[j] += ki * mem[j];
      out[j] += row[j] * qi;
    }
  }
}

void deltanet_seq(float* state, const DeltaSeq& s, float* scratch, int64_t dk, int64_t dv) {
  for (int64_t t = 0; t < s.tokens; ++t) {
    const int64_t o = t * s.qkv_stride;
    const float* g = s.gates + t * s.gate_stride;
    deltanet_step(state, s.q + o, s.k + o, s.v + o, g[1], g[0], s.out + t * s.out_stride, scratch,
                  dk, dv);
  }
}

const KernelSet kScalar = {
    "scalar",      {matmul_f32, matmul_f16, matmul_bf16, matmul_q8, matmul_q4},
    quantize_row,  dot,
    dot_f16,       axpy,
    axpy_f16,      to_f16,
    exp_shifted,   silu,
    silu_mul,      sigmoid_mul,
    deltanet_step, deltanet_seq,
};

}

const KernelSet* scalar_kernels() { return &kScalar; }

}

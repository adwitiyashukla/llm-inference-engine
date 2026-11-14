#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace infer {

void quantize_q8_row(const float* x, int64_t n, int8_t* q, uint16_t* scales) {
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x + b * kQBlock;
    float amax = 0.0f;
    for (int64_t i = 0; i < kQBlock; ++i) amax = std::max(amax, std::fabs(xb[i]));
    uint16_t h = f32_to_f16(amax / 127.0f);
    float d = f16_to_f32(h);
    float inv = d != 0.0f ? 1.0f / d : 0.0f;
    for (int64_t i = 0; i < kQBlock; ++i) {
      int v = static_cast<int>(std::nearbyint(xb[i] * inv));
      q[b * kQBlock + i] = static_cast<int8_t>(std::clamp(v, -127, 127));
    }
    scales[b] = h;
  }
}

void quantize_q4_row(const float* x, int64_t n, int8_t* q, uint16_t* scales) {
  int8_t cand[kQBlock];
  for (int64_t b = 0; b < n / kQBlock; ++b) {
    const float* xb = x + b * kQBlock;
    int8_t* best = q + b * kQBlock;
    float mx = 0.0f;
    for (int64_t i = 0; i < kQBlock; ++i) {
      if (std::fabs(xb[i]) > std::fabs(mx)) mx = xb[i];
    }
    float best_err = std::numeric_limits<float>::infinity();
    float best_d = 0.0f;
    std::fill(best, best + kQBlock, int8_t{0});
    if (mx != 0.0f) {
      for (int step = -9; step <= 9; ++step) {
        float iscale = -(8.0f + 0.1f * static_cast<float>(step)) / mx;
        double sxq = 0.0;
        double sqq = 0.0;
        for (int64_t i = 0; i < kQBlock; ++i) {
          int v = static_cast<int>(std::nearbyint(xb[i] * iscale));
          v = std::clamp(v, -8, 7);
          cand[i] = static_cast<int8_t>(v);
          sxq += static_cast<double>(xb[i]) * v;
          sqq += static_cast<double>(v) * v;
        }
        if (sqq == 0.0) continue;
        float d = f16_to_f32(f32_to_f16(static_cast<float>(sxq / sqq)));
        float err = 0.0f;
        for (int64_t i = 0; i < kQBlock; ++i) {
          float r = xb[i] - d * static_cast<float>(cand[i]);
          err += r * r;
        }
        if (err < best_err) {
          best_err = err;
          best_d = d;
          std::copy(cand, cand + kQBlock, best);
        }
      }
    }
    scales[b] = f32_to_f16(best_d);
  }
}

void init_quant_storage(WType type, int64_t rows, int64_t cols, uint8_t* data, uint16_t* scales) {
  const int64_t n = stored_rows(type, rows);
  std::memset(data, type == WType::Q4 ? 0x88 : 0,
              static_cast<size_t>(n) * wtype_row_bytes(type, cols));
  std::memset(scales, 0, static_cast<size_t>(n * (cols / kQBlock)) * sizeof(uint16_t));
}

void store_quant_row(WType type, int64_t row, int64_t cols, const int8_t* q, const uint16_t* s,
                     uint8_t* data, uint16_t* scales) {
  const int64_t nb = cols / kQBlock;
  const int64_t g = row / kRowGroup;
  const int64_t lane = (row % kRowGroup) * kLaneBytes;
  for (int64_t b = 0; b < nb; ++b) {
    const int64_t tile = g * nb + b;
    scales[tile * kRowGroup + row % kRowGroup] = s[b];
    const int8_t* qb = q + b * kQBlock;
    uint8_t* t = data + tile * quant_tile_bytes(type);
    if (type == WType::Q8) {
      for (int64_t c = 0; c < kQBlock; ++c) {
        t[(c / kLaneBytes) * kVectorBytes + lane + c % kLaneBytes] = static_cast<uint8_t>(qb[c]);
      }
    } else {
      for (int64_t c = 0; c < kQBlock / 2; ++c) {
        uint8_t lo = static_cast<uint8_t>(qb[c] + 8);
        uint8_t hi = static_cast<uint8_t>(qb[c + kQBlock / 2] + 8);
        t[(c / kLaneBytes) * kVectorBytes + lane + c % kLaneBytes] =
            static_cast<uint8_t>(lo | (hi << 4));
      }
    }
  }
}

void load_quant_row(const WeightView& w, int64_t row, int8_t* q, uint16_t* s) {
  const int64_t nb = w.cols / kQBlock;
  const int64_t g = row / kRowGroup;
  const int64_t lane = (row % kRowGroup) * kLaneBytes;
  const uint8_t* data = static_cast<const uint8_t*>(w.data);
  for (int64_t b = 0; b < nb; ++b) {
    const int64_t tile = g * nb + b;
    s[b] = w.scales[tile * kRowGroup + row % kRowGroup];
    const uint8_t* t = data + tile * quant_tile_bytes(w.type);
    int8_t* qb = q + b * kQBlock;
    if (w.type == WType::Q8) {
      for (int64_t c = 0; c < kQBlock; ++c) {
        qb[c] = static_cast<int8_t>(t[(c / kLaneBytes) * kVectorBytes + lane + c % kLaneBytes]);
      }
    } else {
      for (int64_t c = 0; c < kQBlock / 2; ++c) {
        uint8_t v = t[(c / kLaneBytes) * kVectorBytes + lane + c % kLaneBytes];
        qb[c] = static_cast<int8_t>((v & 0x0f) - 8);
        qb[c + kQBlock / 2] = static_cast<int8_t>((v >> 4) - 8);
      }
    }
  }
}

void encode_row(const float* x, int64_t n, WType type, void* data) {
  switch (type) {
    case WType::F32:
      std::memcpy(data, x, static_cast<size_t>(n) * sizeof(float));
      break;
    case WType::F16: {
      uint16_t* p = static_cast<uint16_t*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = f32_to_f16(x[i]);
      break;
    }
    case WType::BF16: {
      uint16_t* p = static_cast<uint16_t*>(data);
      for (int64_t i = 0; i < n; ++i) p[i] = f32_to_bf16(x[i]);
      break;
    }
    case WType::Q8:
    case WType::Q4:
      fail("quantized rows are stored with store_quant_row");
  }
}

void dequantize_row(const WeightView& w, int64_t row, float* out) {
  const int64_t n = w.cols;
  switch (w.type) {
    case WType::F32:
      std::memcpy(out, static_cast<const float*>(w.data) + row * n, static_cast<size_t>(n) * 4);
      break;
    case WType::F16: {
      const uint16_t* p = static_cast<const uint16_t*>(w.data) + row * n;
      for (int64_t i = 0; i < n; ++i) out[i] = f16_to_f32(p[i]);
      break;
    }
    case WType::BF16: {
      const uint16_t* p = static_cast<const uint16_t*>(w.data) + row * n;
      for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_f32(p[i]);
      break;
    }
    case WType::Q8:
    case WType::Q4: {
      const int64_t nb = n / kQBlock;
      thread_local std::vector<int8_t> q;
      thread_local std::vector<uint16_t> s;
      q.resize(static_cast<size_t>(n));
      s.resize(static_cast<size_t>(nb));
      load_quant_row(w, row, q.data(), s.data());
      for (int64_t b = 0; b < nb; ++b) {
        float d = f16_to_f32(s[static_cast<size_t>(b)]);
        for (int64_t i = 0; i < kQBlock; ++i) {
          out[b * kQBlock + i] = d * static_cast<float>(q[static_cast<size_t>(b * kQBlock + i)]);
        }
      }
      break;
    }
  }
}

void make_weight(OwnedWeight& out, const float* values, int64_t rows, int64_t cols, WType type) {
  if (wtype_quantized(type) && cols % kQBlock != 0) fail("quantized weights need cols % 32 == 0");
  const int64_t stored = stored_rows(type, rows);
  const size_t row_bytes = wtype_row_bytes(type, cols);
  out.data.resize(static_cast<size_t>(stored) * row_bytes);
  const int64_t nb = cols / kQBlock;
  if (wtype_quantized(type)) {
    out.scales.resize(static_cast<size_t>(stored * nb));
    init_quant_storage(type, rows, cols, out.data.data(), out.scales.data());
    std::vector<int8_t> q(static_cast<size_t>(cols));
    std::vector<uint16_t> s(static_cast<size_t>(nb));
    for (int64_t r = 0; r < rows; ++r) {
      if (type == WType::Q8) {
        quantize_q8_row(values + r * cols, cols, q.data(), s.data());
      } else {
        quantize_q4_row(values + r * cols, cols, q.data(), s.data());
      }
      store_quant_row(type, r, cols, q.data(), s.data(), out.data.data(), out.scales.data());
    }
  } else {
    for (int64_t r = 0; r < rows; ++r) {
      encode_row(values + r * cols, cols, type, out.data.data() + r * row_bytes);
    }
  }
  out.view.type = type;
  out.view.rows = rows;
  out.view.cols = cols;
  out.view.data = out.data.data();
  out.view.scales = wtype_quantized(type) ? out.scales.data() : nullptr;
}

}

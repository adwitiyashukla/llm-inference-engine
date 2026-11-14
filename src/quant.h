#pragma once

#include <cstdint>

#include "common.h"
#include "kernels.h"

namespace infer {

constexpr int64_t kLaneBytes = 4;
constexpr int64_t kVectorBytes = kRowGroup * kLaneBytes;

inline int64_t quant_tile_bytes(WType type) {
  return type == WType::Q8 ? kRowGroup * kQBlock : kRowGroup * kQBlock / 2;
}

void quantize_q8_row(const float* x, int64_t n, int8_t* q, uint16_t* scales);
void quantize_q4_row(const float* x, int64_t n, int8_t* q, uint16_t* scales);
void init_quant_storage(WType type, int64_t rows, int64_t cols, uint8_t* data, uint16_t* scales);
void store_quant_row(WType type, int64_t row, int64_t cols, const int8_t* q, const uint16_t* s,
                     uint8_t* data, uint16_t* scales);
void load_quant_row(const WeightView& w, int64_t row, int8_t* q, uint16_t* s);
void encode_row(const float* x, int64_t n, WType type, void* data);
void dequantize_row(const WeightView& w, int64_t row, float* out);

struct OwnedWeight {
  WeightView view;
  Buffer<uint8_t> data;
  Buffer<uint16_t> scales;
};

void make_weight(OwnedWeight& out, const float* values, int64_t rows, int64_t cols, WType type);

}

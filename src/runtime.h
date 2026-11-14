#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "common.h"
#include "model.h"
#include "thread_pool.h"

namespace infer {

struct RuntimeOptions {
  int64_t max_batch_tokens = 256;
  int64_t max_sequences = 1;
  int64_t kv_tokens = 4096;
  int64_t block_size = 16;
  bool kv_f16 = true;
  int64_t max_snapshots = 0;
};

class KvPool {
 public:
  void init(int64_t layers, int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t blocks,
            bool f16);
  int32_t allocate();
  void release(int32_t block);
  int64_t free_blocks() const { return static_cast<int64_t>(free_.size()); }
  int64_t blocks() const { return blocks_; }
  int64_t block_size() const { return block_size_; }
  bool f16() const { return f16_; }
  size_t bytes() const;
  uint8_t* key(int64_t layer, int32_t block, int64_t head) {
    return k_[static_cast<size_t>(layer)].data() + offset(block, head);
  }
  uint8_t* value(int64_t layer, int32_t block, int64_t head) {
    return v_[static_cast<size_t>(layer)].data() + offset(block, head);
  }

 private:
  size_t offset(int32_t block, int64_t head) const {
    return (static_cast<size_t>(block) * kv_heads_ + head) * block_size_ * head_dim_ * elem_;
  }
  int64_t layers_ = 0;
  int64_t kv_heads_ = 0;
  int64_t head_dim_ = 0;
  int64_t block_size_ = 16;
  int64_t blocks_ = 0;
  size_t elem_ = 2;
  bool f16_ = true;
  std::vector<Buffer<uint8_t>> k_;
  std::vector<Buffer<uint8_t>> v_;
  std::vector<int32_t> free_;
};

struct Sequence {
  int32_t slot = -1;
  int64_t length = 0;
  std::vector<int32_t> blocks;
  int64_t mtp_length = 0;
  std::vector<int32_t> mtp_blocks;
};

struct BatchRun {
  Sequence* seq = nullptr;
  int64_t first_row = 0;
  int64_t rows = 0;
  int64_t snapshots = 0;
};

struct Batch {
  std::vector<int32_t> tokens;
  std::vector<int64_t> positions;
  std::vector<BatchRun> runs;
  std::vector<int64_t> out_rows;
  void clear() {
    tokens.clear();
    positions.clear();
    runs.clear();
    out_rows.clear();
  }
  int64_t rows() const { return static_cast<int64_t>(tokens.size()); }
  void add_run(Sequence* seq, const int32_t* toks, int64_t count, int64_t first_pos,
               int64_t snapshots, bool output_all);
};

enum class Phase : int {
  Projections,
  LmHead,
  Attention,
  DeltaNet,
  Norms,
  Activations,
  Mtp,
  Other,
  Count
};

struct Projection {
  const WeightView* w;
  float* y;
  int64_t stride;
};

class Runtime {
 public:
  Runtime(const Model& model, ThreadPool& pool, const RuntimeOptions& options);

  const Model& model() const { return model_; }
  const RuntimeOptions& options() const { return opt_; }
  ThreadPool& pool() { return pool_; }

  Sequence* acquire();
  void release(Sequence* seq);
  bool reserve(Sequence& seq, int64_t length, bool mtp);
  void truncate(Sequence& seq, int64_t length) { seq.length = std::min(seq.length, length); }
  void truncate_mtp(Sequence& seq, int64_t length) {
    seq.mtp_length = std::min(seq.mtp_length, length);
  }
  void restore_snapshot(Sequence& seq, int64_t index);
  int64_t free_kv_tokens() const { return main_kv_.free_blocks() * main_kv_.block_size(); }
  size_t state_bytes() const;

  void forward(const Batch& batch, float* hidden_out);
  void forward_mtp(const Batch& batch, const float* hidden_in, float* hidden_out);
  void logits(const float* hidden, int64_t rows, float* out);
  void logits_subset(const float* hidden, int64_t rows, const WeightView& head, float* out,
                     Phase phase = Phase::LmHead);
  void matmul(const float* x, int64_t x_stride, int64_t tokens, const Projection* outs,
              size_t count);

  void set_profiling(bool on) { profiling_ = on; }
  void reset_profile() { profile_.fill(0.0); }
  std::vector<std::pair<std::string, double>> profile() const;

 private:
  class Timer {
   public:
    Timer(Runtime& rt, Phase phase);
    ~Timer();
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

   private:
    Runtime& rt_;
    Phase phase_;
    bool active_;
    double start_;
  };

  void prepare(const Batch& batch);
  void embed_rows(const Batch& batch);
  void norm_rows(const float* in, float* out, const std::vector<float>& w, int64_t rows);
  void decoder_layer(const LayerWeights& layer, const Batch& batch, bool mtp);
  void attention_block(const LayerWeights& layer, const Batch& batch, bool mtp);
  void linear_block(const LayerWeights& layer, const Batch& batch);
  void mlp_block(const LayerWeights& layer, int64_t rows);
  void attend(const Batch& batch, KvPool& kv, int64_t layer, bool mtp, float* q_base,
              int64_t q_stride, int64_t q_head_stride);
  void rope(float* x, int64_t row) const;
  void add_rows(float* dst, const float* src, int64_t rows);
  float* conv_bank(int32_t slot, int64_t bank, int64_t layer);
  float* ssm_bank(int32_t slot, int64_t bank, int64_t layer, int64_t head);
  int64_t snapshot_bank(int32_t slot, int64_t index) const;
  float* conv_state(int32_t slot, int64_t layer);
  float* ssm_state(int32_t slot, int64_t layer, int64_t head);
  float* conv_snapshot(int32_t slot, int64_t index, int64_t layer);
  float* ssm_snapshot(int32_t slot, int64_t index, int64_t layer, int64_t head);

  const Model& model_;
  const ModelConfig& cfg_;
  ThreadPool& pool_;
  RuntimeOptions opt_;
  KvPool main_kv_;
  KvPool mtp_kv_;
  std::vector<Sequence> sequences_;
  std::vector<uint8_t> slot_used_;
  Buffer<float> conv_banks_;
  Buffer<float> ssm_banks_;
  std::vector<int64_t> live_bank_;
  int64_t banks_ = 1;
  size_t conv_state_size_ = 0;
  size_t ssm_head_size_ = 0;
  size_t ssm_layer_size_ = 0;
  int64_t ssm_block_cols_ = 0;
  std::vector<float> inv_freq_;

  Buffer<float> x_;
  Buffer<float> h_;
  Buffer<float> tmp_;
  Buffer<float> proj_;
  Buffer<float> attn_;
  Buffer<float> conv_;
  Buffer<float> act_;
  Buffer<float> mtp_in_;
  Buffer<int8_t> xq_;
  Buffer<float> xd_;
  Buffer<int32_t> xs_;
  Buffer<float> rope_cos_;
  Buffer<float> rope_sin_;
  Buffer<float> gates_;
  Buffer<float> partial_ml_;
  Buffer<float> partial_acc_;
  std::vector<int64_t> row_run_;
  int64_t proj_stride_ = 0;
  int64_t max_cols_ = 0;
  int64_t partial_items_ = 0;
  int64_t max_chunks_ = 0;
  bool profiling_ = false;
  bool timing_ = false;
  std::array<double, static_cast<size_t>(Phase::Count)> profile_{};
};

}

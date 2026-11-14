#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "json.h"
#include "kernels.h"
#include "quant.h"
#include "safetensors.h"
#include "tokenizer.h"

namespace infer {

enum class Arch { Qwen3, Qwen35 };

struct ModelConfig {
  Arch arch = Arch::Qwen3;
  int64_t vocab = 0;
  int64_t hidden = 0;
  int64_t intermediate = 0;
  int64_t layers = 0;
  int64_t heads = 0;
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  int64_t rotary_dim = 0;
  float rope_theta = 10000.0f;
  float rms_eps = 1e-6f;
  bool attn_gate = false;
  bool zero_centered_norm = false;
  bool tie_embeddings = true;
  std::vector<uint8_t> linear;
  int64_t lin_k_heads = 0;
  int64_t lin_v_heads = 0;
  int64_t lin_k_dim = 0;
  int64_t lin_v_dim = 0;
  int64_t conv_kernel = 0;
  int64_t mtp_layers = 0;
  int64_t max_position = 0;

  static ModelConfig from_json(const Json& doc);
  int64_t attn_layers() const;
  int64_t linear_layers() const;
  int64_t conv_dim() const { return 2 * lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim; }
  int64_t q_rows() const { return heads * head_dim * (attn_gate ? 2 : 1); }
  void validate() const;
};

struct AttentionWeights {
  WeightView q;
  WeightView k;
  WeightView v;
  WeightView o;
  std::vector<float> q_norm;
  std::vector<float> k_norm;
};

struct LinearAttentionWeights {
  WeightView qkv;
  WeightView z;
  WeightView b;
  WeightView a;
  WeightView out;
  std::vector<float> conv;
  std::vector<float> neg_exp_a;
  std::vector<float> dt_bias;
  std::vector<float> norm;
};

struct LayerWeights {
  bool linear = false;
  int64_t attn_index = -1;
  int64_t linear_index = -1;
  std::vector<float> input_norm;
  std::vector<float> post_norm;
  AttentionWeights attn;
  LinearAttentionWeights lin;
  WeightView gate;
  WeightView up;
  WeightView down;
};

struct MtpWeights {
  bool present = false;
  WeightView fc;
  std::vector<float> norm_embed;
  std::vector<float> norm_hidden;
  std::vector<float> norm;
  LayerWeights layer;
};

struct ModelWeights {
  WeightView embed;
  WeightView lm_head;
  std::vector<float> final_norm;
  std::vector<LayerWeights> layers;
  MtpWeights mtp;
};

class Model {
 public:
  static std::unique_ptr<Model> load(const std::string& path);
  static std::unique_ptr<Model> random(const ModelConfig& config, WType type, uint64_t seed,
                                       std::unique_ptr<Tokenizer> tokenizer, bool with_mtp);

  const ModelConfig& config() const { return config_; }
  const Tokenizer& tokenizer() const { return *tokenizer_; }
  const ModelWeights& weights() const { return weights_; }
  ChatFormat chat_format() const {
    return config_.arch == Arch::Qwen35 ? ChatFormat::Qwen35 : ChatFormat::Qwen3;
  }
  const std::string& name() const { return name_; }
  const std::string& weight_label() const { return weight_label_; }
  std::vector<int32_t> stop_tokens() const;
  size_t decode_bytes() const;
  size_t parameter_count() const;
  const std::string& config_text() const { return config_text_; }

 private:
  Model() = default;
  void bind(class TensorSource& source);
  void finish();

  ModelConfig config_;
  std::string config_text_;
  std::string name_;
  std::string weight_label_;
  std::unique_ptr<Tokenizer> tokenizer_;
  ModelWeights weights_;
  std::vector<std::unique_ptr<SafeTensors>> files_;
  std::vector<std::unique_ptr<OwnedWeight>> owned_;
};

struct ConvertOptions {
  WType type = WType::Q8;
  int threads = 1;
};

void convert_model(const std::string& hf_dir, const std::string& out_path,
                   const ConvertOptions& options);

}

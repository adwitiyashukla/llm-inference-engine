#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "model.h"
#include "runtime.h"
#include "thread_pool.h"
#include "tokenizer.h"

namespace infer {

struct SamplingParams {
  float temperature = 0.0f;
  int top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0;
};

enum class DraftMode { None, Ngram, Mtp };
enum class FinishReason { None, Stop, Length, Cancelled, Error };

const char* finish_reason_name(FinishReason r);
bool parse_draft_mode(std::string_view s, DraftMode& out);

struct GenerationRequest {
  std::vector<int32_t> prompt;
  SamplingParams sampling;
  int64_t max_tokens = 256;
  std::vector<std::string> stop;
  bool ignore_eos = false;
  DraftMode draft = DraftMode::None;
  int draft_tokens = 0;
};

struct RequestStats {
  int64_t prompt_tokens = 0;
  int64_t generated_tokens = 0;
  int64_t drafted = 0;
  int64_t accepted = 0;
  int64_t decode_steps = 0;
  double queue_seconds = 0.0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
};

struct TokenEvent {
  int64_t request = 0;
  std::string text;
  std::vector<int32_t> tokens;
  bool finished = false;
  FinishReason reason = FinishReason::None;
  std::string error;
  RequestStats stats;
};

using EventCallback = std::function<void(const TokenEvent&)>;

struct EngineOptions {
  int64_t max_sequences = 1;
  int64_t max_batch_tokens = 256;
  int64_t kv_tokens = 4096;
  bool kv_f16 = true;
  int64_t max_draft = 4;
  int64_t draft_vocab = 0;
};

struct GenerationResult {
  std::vector<int32_t> tokens;
  std::string text;
  FinishReason reason = FinishReason::None;
  std::string error;
  RequestStats stats;
};

class Engine {
 public:
  Engine(const Model& model, ThreadPool& pool, const EngineOptions& options);
  ~Engine();

  int64_t submit(GenerationRequest request, EventCallback callback);
  void cancel(int64_t id);
  bool step();
  bool busy() const;
  GenerationResult generate(GenerationRequest request,
                            const std::function<void(const std::string&)>& on_text = {});

  const Model& model() const { return model_; }
  Runtime& runtime() { return *runtime_; }
  const EngineOptions& options() const { return opt_; }

 private:
  struct Active;
  void admit();
  void finish(Active& a, FinishReason reason, const std::string& error = {});
  void emit_token(Active& a, int32_t token, TokenEvent& ev);
  void flush_text(Active& a, TokenEvent& ev);
  void draft_ngram(Active& a);
  void run_mtp(std::vector<Active*>& users, const std::vector<std::vector<float>>& hidden_rows);
  int32_t sample_row(Active& a, const float* logits, int64_t n,
                     std::vector<std::pair<int32_t, float>>* dist);
  void apply_penalties(const Active& a, float* row, int64_t n) const;
  int64_t vocab_limit() const;

  const Model& model_;
  ThreadPool& pool_;
  EngineOptions opt_;
  std::unique_ptr<Runtime> runtime_;
  std::vector<int32_t> stop_tokens_;
  mutable std::mutex mutex_;
  std::deque<std::unique_ptr<Active>> incoming_;
  std::vector<int64_t> cancels_;
  std::vector<std::unique_ptr<Active>> active_;
  std::deque<std::unique_ptr<Active>> waiting_;
  int64_t next_id_ = 1;
  Buffer<float> hidden_;
  Buffer<float> logits_;
  Buffer<float> mtp_hidden_;
  Buffer<float> draft_logits_;
  std::vector<float> dense_q_;
  WeightView draft_head_;
};

}

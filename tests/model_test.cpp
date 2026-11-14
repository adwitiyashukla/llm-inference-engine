#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "check.h"
#include "engine.h"
#include "json.h"
#include "model.h"
#include "runtime.h"
#include "test_models.h"
#include "thread_pool.h"
#include "tokenizer.h"
#include "unicode.h"

using namespace infer;

namespace {

std::vector<int32_t> prompt_tokens(int64_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<int32_t> out;
  for (int64_t i = 0; i < n; ++i) out.push_back(static_cast<int32_t>(rng.next() % 262));
  return out;
}

std::vector<float> prefill_logits(const Model& m, const std::vector<int32_t>& toks, int threads,
                                  int64_t chunk, bool kv_f16 = true) {
  ThreadPool pool(threads);
  RuntimeOptions o;
  o.max_batch_tokens = 64;
  o.kv_tokens = 1024;
  o.kv_f16 = kv_f16;
  Runtime rt(m, pool, o);
  const int64_t H = m.config().hidden;
  const int64_t V = m.config().vocab;
  Sequence* s = rt.acquire();
  std::vector<float> out(toks.size() * static_cast<size_t>(V));
  std::vector<float> hidden(static_cast<size_t>(64 * H));
  for (size_t p = 0; p < toks.size(); p += static_cast<size_t>(chunk)) {
    int64_t n = std::min<int64_t>(chunk, static_cast<int64_t>(toks.size() - p));
    rt.reserve(*s, s->length + n, false);
    Batch b;
    b.add_run(s, toks.data() + p, n, s->length, 0, true);
    rt.forward(b, hidden.data());
    rt.logits(hidden.data(), n, out.data() + p * V);
  }
  return out;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

GenerationResult run(const Model& m, std::vector<int32_t> prompt, DraftMode draft, int draft_tokens,
                     int threads, float temperature = 0.0f, uint64_t seed = 0,
                     int64_t max_tokens = 40) {
  ThreadPool pool(threads);
  EngineOptions o;
  o.kv_tokens = 2048;
  o.max_draft = 4;
  Engine e(m, pool, o);
  GenerationRequest r;
  r.prompt = std::move(prompt);
  r.max_tokens = max_tokens;
  r.ignore_eos = true;
  r.draft = draft;
  r.draft_tokens = draft_tokens;
  r.sampling.temperature = temperature;
  r.sampling.seed = seed;
  return e.generate(std::move(r));
}

}

TEST(incremental_decode_matches_full_prefill_bitwise) {
  for (const ModelConfig& c : {hybrid_config(), dense_config()}) {
    for (WType t : {WType::BF16, WType::Q8, WType::Q4}) {
      auto m = make_model(c, t, false);
      std::vector<int32_t> toks = prompt_tokens(37, 3);
      std::vector<float> full = prefill_logits(*m, toks, 2, 64);
      std::vector<float> one = prefill_logits(*m, toks, 2, 1);
      std::vector<float> mixed = prefill_logits(*m, toks, 2, 5);
      EXPECT(same_bits(full, one));
      EXPECT(same_bits(full, mixed));
    }
  }
}

TEST(results_do_not_depend_on_thread_count) {
  for (const ModelConfig& c : {hybrid_config(), dense_config()}) {
    auto m = make_model(c, WType::Q8, false);
    std::vector<int32_t> toks = prompt_tokens(29, 4);
    std::vector<float> a = prefill_logits(*m, toks, 1, 7);
    for (int threads : {2, 3, 5}) EXPECT(same_bits(a, prefill_logits(*m, toks, threads, 7)));
  }
}

TEST(attention_spans_several_chunks) {
  auto m = make_model(dense_config(), WType::BF16, false);
  std::vector<int32_t> toks = prompt_tokens(600, 8);
  std::vector<float> full = prefill_logits(*m, toks, 3, 64, false);
  std::vector<float> step = prefill_logits(*m, toks, 3, 1, false);
  EXPECT(same_bits(full, step));
}

TEST(quantized_weights_stay_close_to_bf16) {
  for (const ModelConfig& c : {hybrid_config(), dense_config()}) {
    std::vector<int32_t> toks = prompt_tokens(16, 5);
    auto ref = prefill_logits(*make_model(c, WType::F32, false), toks, 2, 16, false);
    for (WType t : {WType::BF16, WType::Q8, WType::Q4}) {
      auto got = prefill_logits(*make_model(c, t, false), toks, 2, 16);
      double err = 0.0;
      double scale = 0.0;
      for (size_t i = 0; i < ref.size(); ++i) {
        err = std::max(err, std::fabs(static_cast<double>(got[i]) - ref[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(ref[i])));
      }
      double tol = t == WType::BF16 ? 0.02 : t == WType::Q8 ? 0.05 : 0.35;
      EXPECT(err < tol * scale);
    }
  }
}

TEST(batched_sequences_match_single_sequences) {
  for (const ModelConfig& c : {hybrid_config(), dense_config()}) {
    auto m = make_model(c, WType::Q8, false);
    std::vector<std::vector<int32_t>> prompts;
    std::vector<GenerationResult> alone;
    for (int i = 0; i < 4; ++i) {
      prompts.push_back(prompt_tokens(5 + 9 * i, 100 + static_cast<uint64_t>(i)));
      alone.push_back(run(*m, prompts.back(), DraftMode::None, 0, 2));
    }
    ThreadPool pool(3);
    EngineOptions o;
    o.max_sequences = 3;
    o.max_batch_tokens = 16;
    o.kv_tokens = 2048;
    Engine e(*m, pool, o);
    std::map<int64_t, std::vector<int32_t>> got;
    std::vector<int64_t> ids;
    for (int i = 0; i < 4; ++i) {
      GenerationRequest r;
      r.prompt = prompts[static_cast<size_t>(i)];
      r.max_tokens = 40;
      r.ignore_eos = true;
      ids.push_back(e.submit(std::move(r), [&got](const TokenEvent& ev) {
        got[ev.request].insert(got[ev.request].end(), ev.tokens.begin(), ev.tokens.end());
      }));
    }
    while (e.busy()) e.step();
    for (int i = 0; i < 4; ++i)
      EXPECT(got[ids[static_cast<size_t>(i)]] == alone[static_cast<size_t>(i)].tokens);
  }
}

TEST(speculative_greedy_is_token_identical) {
  for (const ModelConfig& c : {hybrid_config(), dense_config()}) {
    bool mtp = c.arch == Arch::Qwen35;
    auto m = make_model(c, WType::Q8, mtp);
    std::vector<int32_t> prompt = prompt_tokens(12, 9);
    std::vector<int32_t> repeat(prompt.begin(), prompt.end());
    repeat.insert(repeat.end(), prompt.begin(), prompt.end());
    GenerationResult plain = run(*m, repeat, DraftMode::None, 0, 2, 0.0f, 0, 60);
    for (int k : {1, 3, 4}) {
      GenerationResult ng = run(*m, repeat, DraftMode::Ngram, k, 2, 0.0f, 0, 60);
      EXPECT(ng.tokens == plain.tokens);
      EXPECT(ng.stats.drafted > 0);
      if (mtp) {
        GenerationResult mt = run(*m, repeat, DraftMode::Mtp, k, 2, 0.0f, 0, 60);
        EXPECT(mt.tokens == plain.tokens);
        EXPECT(mt.stats.drafted > 0);
      }
    }
  }
}

TEST(seeded_sampling_is_reproducible) {
  auto m = make_model(hybrid_config(), WType::Q8, true);
  std::vector<int32_t> prompt = prompt_tokens(9, 11);
  for (DraftMode d : {DraftMode::None, DraftMode::Ngram, DraftMode::Mtp}) {
    GenerationResult a = run(*m, prompt, d, 2, 2, 0.8f, 42);
    GenerationResult b = run(*m, prompt, d, 2, 3, 0.8f, 42);
    EXPECT(a.tokens == b.tokens);
    EXPECT(a.tokens.size() == 40);
  }
}

TEST(speculative_sampling_preserves_the_distribution) {
  auto m = make_model(hybrid_config(), WType::Q8, true, 21);
  std::vector<int32_t> prompt = prompt_tokens(6, 12);
  const int trials = 1500;
  std::map<std::pair<int32_t, int32_t>, double> plain;
  std::map<std::pair<int32_t, int32_t>, double> spec;
  ThreadPool pool(2);
  EngineOptions o;
  o.max_sequences = 8;
  o.kv_tokens = 4096;
  o.max_draft = 2;
  Engine e(*m, pool, o);
  for (int mode = 0; mode < 2; ++mode) {
    auto& counts = mode == 0 ? plain : spec;
    for (int i = 0; i < trials; ++i) {
      GenerationRequest r;
      r.prompt = prompt;
      r.max_tokens = 2;
      r.ignore_eos = true;
      r.sampling.temperature = 30.0f;
      r.sampling.top_k = 4;
      r.sampling.seed = static_cast<uint64_t>(1000 * mode + i);
      r.draft = mode == 0 ? DraftMode::None : DraftMode::Mtp;
      r.draft_tokens = 1;
      e.submit(std::move(r), [&counts](const TokenEvent& ev) {
        static std::map<int64_t, std::vector<int32_t>> partial;
        auto& v = partial[ev.request];
        v.insert(v.end(), ev.tokens.begin(), ev.tokens.end());
        if (ev.finished && v.size() == 2) counts[{v[0], v[1]}] += 1.0;
      });
    }
    while (e.busy()) e.step();
  }
  double tv = 0.0;
  std::map<std::pair<int32_t, int32_t>, int> keys;
  for (auto& kv : plain) keys[kv.first] = 1;
  for (auto& kv : spec) keys[kv.first] = 1;
  for (auto& k : keys) tv += std::fabs(plain[k.first] - spec[k.first]) / trials;
  tv *= 0.5;
  std::printf("  total variation between plain and speculative sampling: %.3f over %zu outcomes\n",
              tv, keys.size());
  EXPECT(tv < 0.12);
}

TEST(stop_strings_and_max_tokens) {
  auto m = make_model(dense_config(), WType::Q8, false);
  std::vector<int32_t> prompt = prompt_tokens(8, 13);
  GenerationResult base = run(*m, prompt, DraftMode::None, 0, 2, 0.0f, 0, 30);
  EXPECT(base.reason == FinishReason::Length);
  EXPECT(base.tokens.size() == 30);
  std::string text = base.text;
  DecodedText d = utf8_decode(text);
  EXPECT(d.cps.size() > 4);
  if (d.cps.size() > 4) {
    std::string stop = text.substr(d.offsets[3], d.offsets[4] - d.offsets[3]);
    ThreadPool pool(2);
    EngineOptions o;
    Engine e(*m, pool, o);
    GenerationRequest r;
    r.prompt = prompt;
    r.max_tokens = 30;
    r.ignore_eos = true;
    r.stop = {stop};
    GenerationResult s = e.generate(std::move(r));
    EXPECT(s.reason == FinishReason::Stop);
    EXPECT(s.text == text.substr(0, text.find(stop)));
  }
}

int main() { return check::run_all(); }

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace infer {

const char* finish_reason_name(FinishReason r) {
  switch (r) {
    case FinishReason::None:
      return "none";
    case FinishReason::Stop:
      return "stop";
    case FinishReason::Length:
      return "length";
    case FinishReason::Cancelled:
      return "cancelled";
    case FinishReason::Error:
      return "error";
  }
  return "none";
}

bool parse_draft_mode(std::string_view s, DraftMode& out) {
  if (s == "none") {
    out = DraftMode::None;
  } else if (s == "ngram") {
    out = DraftMode::Ngram;
  } else if (s == "mtp") {
    out = DraftMode::Mtp;
  } else {
    return false;
  }
  return true;
}

struct Engine::Active {
  int64_t id = 0;
  GenerationRequest req;
  EventCallback cb;
  Sequence* seq = nullptr;
  bool decoding = false;
  bool finished = false;
  FinishReason reason = FinishReason::None;
  std::string error;
  std::vector<int32_t> history;
  int64_t prompt_len = 0;
  int64_t prefilled = 0;
  int32_t pending = -1;
  std::vector<int32_t> drafts;
  std::vector<std::vector<std::pair<int32_t, float>>> draft_dists;
  std::vector<int32_t> counts;
  std::vector<int32_t> seen_list;
  std::vector<uint8_t> seen;
  Rng rng;
  std::unique_ptr<StreamDecoder> decoder;
  std::string held;
  int64_t generated = 0;
  double t_submit = 0.0;
  double t_start = 0.0;
  double t_first = 0.0;
  RequestStats stats;
  int64_t out0 = 0;
  int64_t out_count = 0;
  int64_t rows = 0;
  int64_t len_before = 0;
  bool final_prefill = false;
  std::vector<float> mtp_hidden;
  TokenEvent ev;

  bool uses_mtp() const { return req.draft == DraftMode::Mtp; }
  int64_t draft_budget(int64_t max_draft) const {
    int64_t k = req.draft_tokens > 0 ? req.draft_tokens : (req.draft == DraftMode::Mtp ? 2 : 4);
    if (req.draft == DraftMode::None) k = 0;
    k = std::min<int64_t>(k, max_draft);
    k = std::min<int64_t>(k, req.max_tokens - generated - 1);
    return std::max<int64_t>(k, 0);
  }
};

Engine::Engine(const Model& model, ThreadPool& pool, const EngineOptions& options)
    : model_(model), pool_(pool), opt_(options) {
  if (opt_.max_sequences < 1) opt_.max_sequences = 1;
  if (opt_.max_draft < 0) opt_.max_draft = 0;
  opt_.max_batch_tokens =
      std::max(opt_.max_batch_tokens, opt_.max_sequences * (opt_.max_draft + 1));
  RuntimeOptions ro;
  ro.max_batch_tokens = opt_.max_batch_tokens;
  ro.max_sequences = opt_.max_sequences;
  ro.kv_tokens = opt_.kv_tokens;
  ro.kv_f16 = opt_.kv_f16;
  ro.max_snapshots = opt_.max_draft;
  runtime_ = std::make_unique<Runtime>(model_, pool_, ro);
  stop_tokens_ = model_.stop_tokens();
  const int64_t H = model_.config().hidden;
  hidden_.resize(static_cast<size_t>(opt_.max_batch_tokens * H));
  mtp_hidden_.resize(static_cast<size_t>(opt_.max_batch_tokens * H));
  const int64_t logit_rows = opt_.max_sequences * (opt_.max_draft + 2);
  logits_.resize(static_cast<size_t>(logit_rows * model_.config().vocab));
  draft_head_ = model_.weights().lm_head;
  if (opt_.draft_vocab > 0 && opt_.draft_vocab < draft_head_.rows)
    draft_head_.rows = opt_.draft_vocab;
  if (model_.weights().mtp.present)
    draft_logits_.resize(static_cast<size_t>(opt_.max_sequences * draft_head_.rows));
}

Engine::~Engine() {
  for (auto& a : active_) runtime_->release(a->seq);
}

int64_t Engine::vocab_limit() const {
  return std::min<int64_t>(model_.config().vocab, model_.tokenizer().size());
}

int64_t Engine::submit(GenerationRequest request, EventCallback callback) {
  auto a = std::make_unique<Active>();
  a->req = std::move(request);
  a->cb = std::move(callback);
  a->t_submit = now_seconds();
  std::lock_guard<std::mutex> lock(mutex_);
  a->id = next_id_++;
  int64_t id = a->id;
  incoming_.push_back(std::move(a));
  return id;
}

void Engine::cancel(int64_t id) {
  std::lock_guard<std::mutex> lock(mutex_);
  cancels_.push_back(id);
}

bool Engine::busy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return !incoming_.empty() || !active_.empty() || !waiting_.empty();
}

void Engine::admit() {
  while (!waiting_.empty()) {
    Active& a = *waiting_.front();
    if (a.req.prompt.empty()) {
      a.finished = true;
      a.reason = FinishReason::Error;
      a.error = "empty prompt";
      a.ev.request = a.id;
      a.ev.finished = true;
      a.ev.reason = a.reason;
      a.ev.error = a.error;
      if (a.cb) a.cb(a.ev);
      waiting_.pop_front();
      continue;
    }
    if (static_cast<int64_t>(a.req.prompt.size()) + 1 > opt_.kv_tokens) {
      a.finished = true;
      a.reason = FinishReason::Error;
      a.error = "prompt does not fit in the kv cache";
      a.ev.request = a.id;
      a.ev.finished = true;
      a.ev.reason = a.reason;
      a.ev.error = a.error;
      if (a.cb) a.cb(a.ev);
      waiting_.pop_front();
      continue;
    }
    Sequence* seq = runtime_->acquire();
    if (seq == nullptr) break;
    std::unique_ptr<Active> p = std::move(waiting_.front());
    waiting_.pop_front();
    p->seq = seq;
    p->t_start = now_seconds();
    p->history = p->req.prompt;
    p->prompt_len = static_cast<int64_t>(p->history.size());
    p->rng.reseed(p->req.sampling.seed);
    p->decoder = std::make_unique<StreamDecoder>(model_.tokenizer());
    p->counts.assign(static_cast<size_t>(model_.config().vocab), 0);
    p->seen.assign(static_cast<size_t>(model_.config().vocab), 0);
    for (int32_t t : p->history) {
      if (t >= 0 && t < model_.config().vocab && !p->seen[static_cast<size_t>(t)]) {
        p->seen[static_cast<size_t>(t)] = 1;
        p->seen_list.push_back(t);
      }
    }
    p->stats.prompt_tokens = p->prompt_len;
    if (p->req.draft == DraftMode::Mtp && !model_.weights().mtp.present)
      p->req.draft = DraftMode::None;
    active_.push_back(std::move(p));
  }
}

void Engine::apply_penalties(const Active& a, float* row, int64_t n) const {
  const SamplingParams& s = a.req.sampling;
  if (s.repetition_penalty != 1.0f) {
    for (int32_t t : a.seen_list) {
      if (t >= n) continue;
      float& v = row[t];
      v = v > 0.0f ? v / s.repetition_penalty : v * s.repetition_penalty;
    }
  }
  if (s.presence_penalty != 0.0f || s.frequency_penalty != 0.0f) {
    for (int32_t t : a.seen_list) {
      if (t >= n) continue;
      int32_t c = a.counts[static_cast<size_t>(t)];
      if (c > 0) row[t] -= s.frequency_penalty * static_cast<float>(c) + s.presence_penalty;
    }
  }
}

int32_t Engine::sample_row(Active& a, const float* logits, int64_t n,
                           std::vector<std::pair<int32_t, float>>* dist) {
  const int64_t V = n;
  const SamplingParams& s = a.req.sampling;
  const bool penalties =
      s.repetition_penalty != 1.0f || s.presence_penalty != 0.0f || s.frequency_penalty != 0.0f;
  if (s.temperature <= 0.0f && !penalties) {
    int32_t best = static_cast<int32_t>(std::max_element(logits, logits + V) - logits);
    if (dist != nullptr) dist->assign(1, {best, 1.0f});
    return best;
  }
  thread_local std::vector<float> row;
  row.assign(logits, logits + V);
  apply_penalties(a, row.data(), V);
  if (s.temperature <= 0.0f) {
    int32_t best = static_cast<int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
    if (dist != nullptr) dist->assign(1, {best, 1.0f});
    return best;
  }
  const float inv_t = 1.0f / s.temperature;
  for (float& v : row) v *= inv_t;
  if ((s.top_k <= 0 || s.top_k >= V) && s.top_p >= 1.0f && s.min_p <= 0.0f) {
    const float mx = *std::max_element(row.begin(), row.end());
    kernels().exp_shifted(row.data(), V, mx);
    double total = 0.0;
    for (float w : row) total += w;
    if (dist != nullptr) {
      dist->clear();
      for (int64_t i = 0; i < V; ++i) {
        const float w = row[static_cast<size_t>(i)];
        if (w > 0.0f) dist->push_back({static_cast<int32_t>(i), static_cast<float>(w / total)});
      }
    }
    double u = static_cast<double>(a.rng.uniform()) * total;
    int32_t last = 0;
    for (int64_t i = 0; i < V; ++i) {
      const float w = row[static_cast<size_t>(i)];
      if (w <= 0.0f) continue;
      last = static_cast<int32_t>(i);
      u -= w;
      if (u < 0.0) return last;
    }
    return last;
  }
  thread_local std::vector<int32_t> order;
  order.resize(static_cast<size_t>(V));
  for (int64_t i = 0; i < V; ++i) order[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  auto by_logit = [&](int32_t x, int32_t y) {
    return row[x] > row[y] || (row[x] == row[y] && x < y);
  };
  int64_t limit = s.top_k > 0 ? std::min<int64_t>(s.top_k, V) : V;
  int64_t sorted = std::min<int64_t>(limit, 1024);
  std::partial_sort(order.begin(), order.begin() + sorted, order.end(), by_logit);
  const float mx = row[order[0]];
  double z = 0.0;
  if (s.top_k > 0) {
    for (int64_t i = 0; i < limit; ++i)
      z += std::exp(static_cast<double>(row[order[static_cast<size_t>(i)]] - mx));
  } else {
    for (int64_t i = 0; i < V; ++i)
      z += std::exp(static_cast<double>(row[static_cast<size_t>(i)] - mx));
  }
  int64_t keep = limit;
  if (s.top_p < 1.0f) {
    double cum = 0.0;
    int64_t i = 0;
    for (;; ++i) {
      if (i == sorted && sorted < limit) {
        std::sort(order.begin() + sorted, order.begin() + limit, by_logit);
        sorted = limit;
      }
      if (i >= limit) break;
      if (cum >= s.top_p) break;
      cum += std::exp(static_cast<double>(row[order[static_cast<size_t>(i)]] - mx)) / z;
    }
    keep = std::max<int64_t>(1, i);
  }
  if (keep > sorted) {
    std::sort(order.begin() + sorted, order.begin() + keep, by_logit);
    sorted = keep;
  }
  if (s.min_p > 0.0f) {
    double floor = std::log(static_cast<double>(s.min_p));
    int64_t i = 1;
    while (i < keep && static_cast<double>(row[order[static_cast<size_t>(i)]] - mx) >= floor) ++i;
    keep = i;
  }
  thread_local std::vector<double> probs;
  probs.resize(static_cast<size_t>(keep));
  double total = 0.0;
  for (int64_t i = 0; i < keep; ++i) {
    probs[static_cast<size_t>(i)] =
        std::exp(static_cast<double>(row[order[static_cast<size_t>(i)]] - mx));
    total += probs[static_cast<size_t>(i)];
  }
  if (dist != nullptr) {
    dist->clear();
    for (int64_t i = 0; i < keep; ++i) {
      dist->push_back({order[static_cast<size_t>(i)],
                       static_cast<float>(probs[static_cast<size_t>(i)] / total)});
    }
  }
  double u = static_cast<double>(a.rng.uniform()) * total;
  for (int64_t i = 0; i < keep; ++i) {
    u -= probs[static_cast<size_t>(i)];
    if (u < 0.0) return order[static_cast<size_t>(i)];
  }
  return order[static_cast<size_t>(keep - 1)];
}

static size_t stop_prefix_hold(const std::string& text, const std::vector<std::string>& stops) {
  size_t hold = 0;
  for (const std::string& s : stops) {
    size_t max_len = std::min(text.size(), s.size() - 1);
    for (size_t len = max_len; len > hold; --len) {
      if (text.compare(text.size() - len, len, s, 0, len) == 0) {
        hold = len;
        break;
      }
    }
  }
  return hold;
}

void Engine::flush_text(Active& a, TokenEvent& ev) {
  a.held += a.decoder->flush();
  ev.text += a.held;
  a.held.clear();
}

void Engine::finish(Active& a, FinishReason reason, const std::string& error) {
  if (a.finished) return;
  a.finished = true;
  a.reason = reason;
  a.error = error;
  if (reason != FinishReason::Error) flush_text(a, a.ev);
  double now = now_seconds();
  a.stats.generated_tokens = a.generated;
  a.stats.queue_seconds = a.t_start > 0.0 ? a.t_start - a.t_submit : now - a.t_submit;
  if (a.t_first > 0.0) {
    a.stats.prefill_seconds = a.t_first - a.t_start;
    a.stats.decode_seconds = now - a.t_first;
  } else if (a.t_start > 0.0) {
    a.stats.prefill_seconds = now - a.t_start;
  }
  a.ev.finished = true;
  a.ev.reason = reason;
  a.ev.error = error;
  a.ev.stats = a.stats;
}

void Engine::emit_token(Active& a, int32_t token, TokenEvent& ev) {
  if (a.finished) return;
  a.history.push_back(token);
  ++a.generated;
  if (token >= 0 && token < model_.config().vocab) {
    ++a.counts[static_cast<size_t>(token)];
    if (!a.seen[static_cast<size_t>(token)]) {
      a.seen[static_cast<size_t>(token)] = 1;
      a.seen_list.push_back(token);
    }
  }
  bool eos = std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
  if (eos && !a.req.ignore_eos) {
    finish(a, FinishReason::Stop);
    return;
  }
  ev.tokens.push_back(token);
  a.held += a.decoder->push(token);
  size_t best = std::string::npos;
  for (const std::string& s : a.req.stop) {
    if (s.empty()) continue;
    size_t p = a.held.find(s);
    if (p != std::string::npos && (best == std::string::npos || p < best)) best = p;
  }
  if (best != std::string::npos) {
    ev.text += a.held.substr(0, best);
    a.held.clear();
    a.decoder = std::make_unique<StreamDecoder>(model_.tokenizer());
    finish(a, FinishReason::Stop);
    return;
  }
  size_t hold = stop_prefix_hold(a.held, a.req.stop);
  ev.text += a.held.substr(0, a.held.size() - hold);
  a.held.erase(0, a.held.size() - hold);
  if (a.generated >= a.req.max_tokens) finish(a, FinishReason::Length);
}

void Engine::draft_ngram(Active& a) {
  a.drafts.clear();
  a.draft_dists.clear();
  const int64_t k = a.draft_budget(opt_.max_draft);
  if (k <= 0) return;
  const std::vector<int32_t>& h = a.history;
  const int64_t n = static_cast<int64_t>(h.size());
  for (int64_t ng = std::min<int64_t>(3, n - 1); ng >= 2; --ng) {
    for (int64_t start = n - ng - 1; start >= 0; --start) {
      if (!std::equal(h.begin() + start, h.begin() + start + ng, h.end() - ng)) continue;
      for (int64_t j = start + ng; j < n && static_cast<int64_t>(a.drafts.size()) < k; ++j) {
        a.drafts.push_back(h[static_cast<size_t>(j)]);
      }
      if (!a.drafts.empty()) return;
    }
  }
}

void Engine::run_mtp(std::vector<Active*>& users,
                     const std::vector<std::vector<float>>& hidden_rows) {
  const int64_t H = model_.config().hidden;
  const int64_t V = draft_head_.rows;
  Batch batch;
  std::vector<float> hidden_in;
  std::vector<int32_t> tokens;
  std::vector<Active*> drafting;
  for (size_t u = 0; u < users.size(); ++u) {
    Active& a = *users[u];
    const std::vector<float>& rows = hidden_rows[u];
    const int64_t count = static_cast<int64_t>(rows.size()) / H;
    if (count == 0) continue;
    const int64_t first = a.seq->length - count;
    tokens.assign(a.history.begin() + first + 1, a.history.begin() + first + 1 + count);
    runtime_->truncate_mtp(*a.seq, first);
    if (!runtime_->reserve(*a.seq, first + count + opt_.max_draft + 1, true)) {
      a.req.draft = DraftMode::None;
      continue;
    }
    batch.add_run(a.seq, tokens.data(), count, first, 0, false);
    hidden_in.insert(hidden_in.end(), rows.begin(), rows.end());
    drafting.push_back(&a);
  }
  if (batch.rows() == 0) return;
  runtime_->forward_mtp(batch, hidden_in.data(), mtp_hidden_.data());
  std::vector<Active*> chain;
  for (size_t i = 0; i < drafting.size(); ++i) {
    Active& a = *drafting[i];
    a.mtp_hidden.assign(mtp_hidden_.data() + i * H, mtp_hidden_.data() + (i + 1) * H);
    a.drafts.clear();
    a.draft_dists.clear();
    if (a.decoding && !a.finished && a.draft_budget(opt_.max_draft) > 0) chain.push_back(&a);
  }
  const int64_t sample_width = std::min(V, vocab_limit());
  while (!chain.empty()) {
    const int64_t n = static_cast<int64_t>(chain.size());
    std::vector<float> h(static_cast<size_t>(n * H));
    for (int64_t i = 0; i < n; ++i) {
      std::copy(chain[static_cast<size_t>(i)]->mtp_hidden.begin(),
                chain[static_cast<size_t>(i)]->mtp_hidden.end(), h.begin() + i * H);
    }
    float* lg = draft_logits_.data();
    runtime_->logits_subset(h.data(), n, draft_head_, lg, Phase::Mtp);
    std::vector<Active*> next;
    Batch step;
    for (int64_t i = 0; i < n; ++i) {
      Active& a = *chain[static_cast<size_t>(i)];
      std::vector<std::pair<int32_t, float>> dist;
      int32_t d = sample_row(a, lg + i * V, sample_width, &dist);
      a.drafts.push_back(d);
      a.draft_dists.push_back(std::move(dist));
      if (static_cast<int64_t>(a.drafts.size()) < a.draft_budget(opt_.max_draft))
        next.push_back(&a);
    }
    if (next.empty()) break;
    std::vector<float> hin;
    for (Active* a : next) {
      int32_t tok = a->drafts.back();
      step.add_run(a->seq, &tok, 1, a->seq->mtp_length, 0, true);
      hin.insert(hin.end(), a->mtp_hidden.begin(), a->mtp_hidden.end());
    }
    runtime_->forward_mtp(step, hin.data(), mtp_hidden_.data());
    for (size_t i = 0; i < next.size(); ++i) {
      next[i]->mtp_hidden.assign(mtp_hidden_.data() + i * H, mtp_hidden_.data() + (i + 1) * H);
    }
    chain = next;
  }
}

bool Engine::step() {
  std::vector<int64_t> cancels;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!incoming_.empty()) {
      waiting_.push_back(std::move(incoming_.front()));
      incoming_.pop_front();
    }
    cancels.swap(cancels_);
  }
  for (int64_t id : cancels) {
    for (auto& a : active_) {
      if (a->id == id) finish(*a, FinishReason::Cancelled);
    }
    for (auto it = waiting_.begin(); it != waiting_.end(); ++it) {
      if ((*it)->id == id) {
        Active& a = **it;
        a.ev.request = a.id;
        finish(a, FinishReason::Cancelled);
        if (a.cb) a.cb(a.ev);
        waiting_.erase(it);
        break;
      }
    }
  }
  admit();
  if (active_.empty()) return false;

  const int64_t H = model_.config().hidden;
  const int64_t V = model_.config().vocab;
  const bool has_linear = model_.config().linear_layers() > 0;
  Batch batch;
  int64_t budget = opt_.max_batch_tokens;
  for (auto& p : active_) {
    Active& a = *p;
    a.ev = TokenEvent();
    a.ev.request = a.id;
    a.rows = 0;
    if (a.finished || !a.decoding) continue;
    std::vector<int32_t> toks;
    toks.push_back(a.pending);
    toks.insert(toks.end(), a.drafts.begin(), a.drafts.end());
    const int64_t rows = static_cast<int64_t>(toks.size());
    if (!runtime_->reserve(*a.seq, a.seq->length + rows, false)) {
      finish(a, FinishReason::Length, "kv cache is full");
      continue;
    }
    a.len_before = a.seq->length;
    a.out0 = static_cast<int64_t>(batch.out_rows.size());
    a.rows = rows;
    a.out_count = rows;
    batch.add_run(a.seq, toks.data(), rows, a.seq->length, has_linear ? rows - 1 : 0, true);
    budget -= rows;
  }
  for (auto& p : active_) {
    Active& a = *p;
    if (a.finished || a.decoding || budget <= 0) continue;
    const int64_t n = std::min(a.prompt_len - a.prefilled, budget);
    if (!runtime_->reserve(*a.seq, a.seq->length + n + 1, false)) {
      finish(a, FinishReason::Length, "kv cache is full");
      continue;
    }
    a.len_before = a.seq->length;
    a.out0 = static_cast<int64_t>(batch.out_rows.size());
    a.rows = n;
    a.out_count = a.uses_mtp() ? n : 1;
    a.final_prefill = a.prefilled + n == a.prompt_len;
    batch.add_run(a.seq, a.history.data() + a.prefilled, n, a.seq->length, 0, a.uses_mtp());
    budget -= n;
  }

  if (batch.rows() > 0) {
    try {
      runtime_->forward(batch, hidden_.data());
    } catch (const std::exception& e) {
      for (auto& p : active_) {
        if (p->rows > 0) finish(*p, FinishReason::Error, e.what());
      }
      batch.clear();
    }
  }

  std::vector<int64_t> logit_rows;
  std::vector<int64_t> logit_base(active_.size(), -1);
  for (size_t i = 0; i < active_.size(); ++i) {
    Active& a = *active_[i];
    if (a.finished || a.rows == 0) continue;
    if (a.decoding) {
      logit_base[i] = static_cast<int64_t>(logit_rows.size());
      for (int64_t r = 0; r < a.rows; ++r) logit_rows.push_back(a.out0 + r);
    } else if (a.final_prefill) {
      logit_base[i] = static_cast<int64_t>(logit_rows.size());
      logit_rows.push_back(a.out0 + a.out_count - 1);
    }
  }
  if (!logit_rows.empty()) {
    std::vector<float> gathered(logit_rows.size() * static_cast<size_t>(H));
    for (size_t k = 0; k < logit_rows.size(); ++k) {
      std::copy(hidden_.data() + logit_rows[k] * H, hidden_.data() + (logit_rows[k] + 1) * H,
                gathered.begin() + static_cast<int64_t>(k) * H);
    }
    const int64_t total = static_cast<int64_t>(logit_rows.size());
    for (int64_t r = 0; r < total; r += opt_.max_batch_tokens) {
      int64_t n = std::min(opt_.max_batch_tokens, total - r);
      runtime_->logits(gathered.data() + r * H, n, logits_.data() + r * V);
    }
  }

  std::vector<Active*> mtp_users;
  std::vector<std::vector<float>> mtp_rows;
  for (size_t i = 0; i < active_.size(); ++i) {
    Active& a = *active_[i];
    if (a.finished || a.rows == 0) continue;
    if (!a.decoding) {
      std::vector<float> rows;
      if (a.uses_mtp())
        rows.assign(hidden_.data() + a.out0 * H, hidden_.data() + (a.out0 + a.rows) * H);
      a.prefilled += a.rows;
      if (a.final_prefill) {
        int32_t tok = sample_row(a, logits_.data() + logit_base[i] * V, vocab_limit(), nullptr);
        a.t_first = now_seconds();
        a.decoding = true;
        a.pending = tok;
        emit_token(a, tok, a.ev);
      }
      if (a.uses_mtp() && !a.finished) {
        mtp_users.push_back(&a);
        mtp_rows.push_back(std::move(rows));
      }
      continue;
    }
    const int64_t k = a.rows - 1;
    int64_t accepted = 0;
    int32_t next = -1;
    const float* base = logits_.data() + logit_base[i] * V;
    if (a.req.sampling.temperature <= 0.0f) {
      for (int64_t j = 0; j <= k; ++j) {
        int32_t pred = sample_row(a, base + j * V, vocab_limit(), nullptr);
        if (j < k && pred == a.drafts[static_cast<size_t>(j)]) {
          ++accepted;
          emit_token(a, pred, a.ev);
          if (a.finished) break;
          continue;
        }
        next = pred;
        break;
      }
    } else {
      std::vector<std::pair<int32_t, float>> p;
      for (int64_t j = 0; j <= k; ++j) {
        int32_t sampled = sample_row(a, base + j * V, vocab_limit(), &p);
        if (j == k) {
          next = sampled;
          break;
        }
        const int32_t d = a.drafts[static_cast<size_t>(j)];
        float pd = 0.0f;
        for (const auto& e : p) {
          if (e.first == d) pd = e.second;
        }
        float qd = 1.0f;
        const std::vector<std::pair<int32_t, float>>* q = nullptr;
        if (j < static_cast<int64_t>(a.draft_dists.size()) &&
            !a.draft_dists[static_cast<size_t>(j)].empty()) {
          q = &a.draft_dists[static_cast<size_t>(j)];
          qd = 0.0f;
          for (const auto& e : *q) {
            if (e.first == d) qd = e.second;
          }
        }
        if (qd > 0.0f && a.rng.uniform() < std::min(1.0f, pd / qd)) {
          ++accepted;
          emit_token(a, d, a.ev);
          if (a.finished) break;
          continue;
        }
        if (dense_q_.empty()) dense_q_.assign(static_cast<size_t>(model_.config().vocab), 0.0f);
        if (q != nullptr) {
          for (const auto& f : *q) dense_q_[static_cast<size_t>(f.first)] = f.second;
        } else {
          dense_q_[static_cast<size_t>(d)] = 1.0f;
        }
        std::vector<std::pair<int32_t, double>> residual;
        double total = 0.0;
        for (const auto& e : p) {
          const double qv = dense_q_[static_cast<size_t>(e.first)];
          double r = std::max(0.0, static_cast<double>(e.second) - qv);
          if (r > 0.0) {
            residual.push_back({e.first, r});
            total += r;
          }
        }
        if (q != nullptr) {
          for (const auto& f : *q) dense_q_[static_cast<size_t>(f.first)] = 0.0f;
        } else {
          dense_q_[static_cast<size_t>(d)] = 0.0f;
        }
        if (total <= 0.0) {
          next = sampled;
        } else {
          double u = static_cast<double>(a.rng.uniform()) * total;
          next = residual.back().first;
          for (const auto& e : residual) {
            u -= e.second;
            if (u < 0.0) {
              next = e.first;
              break;
            }
          }
        }
        break;
      }
    }
    a.stats.drafted += k;
    a.stats.accepted += accepted;
    ++a.stats.decode_steps;
    if (a.finished) continue;
    runtime_->truncate(*a.seq, a.len_before + accepted + 1);
    if (accepted < k && has_linear) runtime_->restore_snapshot(*a.seq, accepted);
    if (a.uses_mtp()) {
      mtp_users.push_back(&a);
      mtp_rows.emplace_back(hidden_.data() + a.out0 * H,
                            hidden_.data() + (a.out0 + accepted + 1) * H);
    }
    a.pending = next;
    emit_token(a, next, a.ev);
  }

  if (!mtp_users.empty()) {
    std::vector<Active*> live;
    std::vector<std::vector<float>> rows;
    for (size_t i = 0; i < mtp_users.size(); ++i) {
      if (mtp_users[i]->finished) continue;
      live.push_back(mtp_users[i]);
      rows.push_back(std::move(mtp_rows[i]));
    }
    try {
      run_mtp(live, rows);
    } catch (const std::exception& e) {
      for (Active* a : live) finish(*a, FinishReason::Error, e.what());
    }
  }
  for (auto& p : active_) {
    Active& a = *p;
    if (!a.finished && a.decoding && a.req.draft == DraftMode::Ngram) draft_ngram(a);
    if (!a.finished && a.decoding && a.req.draft == DraftMode::None) a.drafts.clear();
  }
  for (auto& p : active_) {
    Active& a = *p;
    if (a.cb && (a.finished || !a.ev.text.empty() || !a.ev.tokens.empty())) a.cb(a.ev);
  }
  for (auto it = active_.begin(); it != active_.end();) {
    if ((*it)->finished) {
      runtime_->release((*it)->seq);
      it = active_.erase(it);
    } else {
      ++it;
    }
  }
  return true;
}

GenerationResult Engine::generate(GenerationRequest request,
                                  const std::function<void(const std::string&)>& on_text) {
  GenerationResult result;
  bool done = false;
  submit(std::move(request), [&](const TokenEvent& ev) {
    result.text += ev.text;
    result.tokens.insert(result.tokens.end(), ev.tokens.begin(), ev.tokens.end());
    if (on_text && !ev.text.empty()) on_text(ev.text);
    if (ev.finished) {
      done = true;
      result.reason = ev.reason;
      result.error = ev.error;
      result.stats = ev.stats;
    }
  });
  while (!done) step();
  return result;
}

}

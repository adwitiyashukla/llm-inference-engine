#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "engine.h"
#include "json.h"
#include "kernels.h"
#include "model.h"
#include "runtime.h"
#include "safetensors.h"
#include "server.h"
#include "thread_pool.h"
#include "tokenizer.h"
#include "unicode.h"

#if defined(_WIN32)
#include <windows.h>

#include <shellapi.h>
#endif

using namespace infer;

namespace {

std::atomic<bool> g_interrupted{false};

void on_sigint(int) { g_interrupted.store(true); }

class Args {
 public:
  Args(const std::vector<std::string>& argv, size_t start, const std::set<std::string>& flags) {
    for (size_t i = start; i < argv.size(); ++i) {
      const std::string& a = argv[i];
      if (a.size() > 1 && a[0] == '-' &&
          !(a.size() > 1 && std::isdigit(static_cast<unsigned char>(a[1])))) {
        std::string key = a.substr(a[1] == '-' ? 2 : 1);
        key = alias(key);
        if (flags.count(key)) {
          flags_.insert(key);
        } else if (i + 1 < argv.size()) {
          options_[key] = argv[++i];
        } else {
          fail("option --" + key + " needs a value");
        }
      } else {
        positional_.push_back(a);
      }
    }
  }
  static std::string alias(const std::string& k) {
    static const std::map<std::string, std::string> aliases = {
        {"p", "prompt"}, {"n", "max-tokens"}, {"t", "threads"}, {"m", "model"}, {"f", "file"}};
    auto it = aliases.find(k);
    return it == aliases.end() ? k : it->second;
  }
  bool has(const std::string& k) const { return flags_.count(k) > 0 || options_.count(k) > 0; }
  std::string get(const std::string& k, const std::string& fallback = {}) const {
    auto it = options_.find(k);
    return it == options_.end() ? fallback : it->second;
  }
  int64_t get_int(const std::string& k, int64_t fallback) const {
    auto it = options_.find(k);
    return it == options_.end() ? fallback : std::stoll(it->second);
  }
  double get_num(const std::string& k, double fallback) const {
    auto it = options_.find(k);
    return it == options_.end() ? fallback : std::stod(it->second);
  }
  std::vector<int64_t> get_list(const std::string& k, int64_t fallback) const {
    std::vector<int64_t> out;
    std::stringstream ss(get(k, std::to_string(fallback)));
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) out.push_back(std::stoll(item));
    }
    return out;
  }
  const std::vector<std::string>& positional() const { return positional_; }
  std::string arg(size_t i, const char* what) const {
    if (i >= positional_.size()) fail(std::string("missing ") + what);
    return positional_[i];
  }

 private:
  std::vector<std::string> positional_;
  std::map<std::string, std::string> options_;
  std::set<std::string> flags_;
};

const std::set<std::string> kFlags = {"chat",   "think", "no-think",   "stats",
                                      "static", "raw",   "ignore-eos", "sweep",
                                      "quiet",  "help",  "ids",        "profile"};

void setup_runtime(const Args& a) {
  if (a.has("kernels")) use_kernels(a.get("kernels"));
}

int threads_of(const Args& a) {
  return static_cast<int>(a.get_int("threads", default_thread_count()));
}

std::string human_bytes(double b) {
  const char* units[] = {"B", "KB", "MB", "GB"};
  int u = 0;
  while (b >= 1000.0 && u < 3) {
    b /= 1000.0;
    ++u;
  }
  return strf("%.1f %s", b, units[u]);
}

SamplingParams sampling_of(const Args& a, float default_temp) {
  SamplingParams s;
  s.temperature = static_cast<float>(a.get_num("temp", default_temp));
  s.top_k = static_cast<int>(a.get_int("top-k", 0));
  s.top_p = static_cast<float>(a.get_num("top-p", 1.0));
  s.min_p = static_cast<float>(a.get_num("min-p", 0.0));
  s.repetition_penalty = static_cast<float>(a.get_num("repeat-penalty", 1.0));
  s.presence_penalty = static_cast<float>(a.get_num("presence-penalty", 0.0));
  s.seed = static_cast<uint64_t>(a.get_int("seed", 0));
  return s;
}

DraftMode draft_of(const Args& a) {
  DraftMode d = DraftMode::None;
  if (a.has("draft") && !parse_draft_mode(a.get("draft"), d))
    fail("--draft must be none, ngram or mtp");
  return d;
}

void check_draft(const Args& a, const Model& m) {
  if (draft_of(a) == DraftMode::Mtp && !m.weights().mtp.present) {
    std::fprintf(stderr,
                 "this model has no multi-token prediction head, decoding without drafts\n");
  }
}

EngineOptions engine_options(const Args& a, int64_t sequences) {
  EngineOptions o;
  o.max_sequences = sequences;
  o.kv_tokens = a.get_int("ctx", 4096) * sequences;
  o.max_batch_tokens = a.get_int("batch-tokens", 256);
  o.kv_f16 = a.get("kv", "f16") != "f32";
  o.max_draft = std::max<int64_t>(4, a.get_int("draft-tokens", 0));
  o.draft_vocab = a.get_int("draft-vocab", 32768);
  return o;
}

int thinking_of(const Args& a) {
  if (a.has("think")) return 1;
  if (a.has("no-think")) return 0;
  return -1;
}

std::unique_ptr<Model> load_model(const Args& a, size_t index) {
  std::string path = a.arg(index, "model path");
  double t0 = now_seconds();
  auto m = Model::load(path);
  if (!a.has("quiet")) {
    std::fprintf(stderr, "loaded %s (%s, %s weights) in %.2fs\n", m->name().c_str(),
                 m->config().arch == Arch::Qwen35 ? "qwen3.5 hybrid" : "qwen3",
                 m->weight_label().c_str(), now_seconds() - t0);
  }
  return m;
}

void print_stats(const RequestStats& st) {
  double prefill =
      st.prefill_seconds > 0.0 ? static_cast<double>(st.prompt_tokens) / st.prefill_seconds : 0.0;
  double decode =
      st.decode_seconds > 0.0
          ? static_cast<double>(std::max<int64_t>(st.generated_tokens - 1, 0)) / st.decode_seconds
          : 0.0;
  std::fprintf(stderr, "\nprompt %lld tokens, %.1f tok/s | generated %lld tokens, %.1f tok/s",
               static_cast<long long>(st.prompt_tokens), prefill,
               static_cast<long long>(st.generated_tokens), decode);
  if (st.drafted > 0) {
    std::fprintf(stderr, " | drafts accepted %lld/%lld (%.0f%%), %.2f tokens per step",
                 static_cast<long long>(st.accepted), static_cast<long long>(st.drafted),
                 100.0 * static_cast<double>(st.accepted) / static_cast<double>(st.drafted),
                 st.decode_steps > 0 ? static_cast<double>(st.generated_tokens - 1) /
                                           static_cast<double>(st.decode_steps)
                                     : 0.0);
  }
  std::fprintf(stderr, " | kernels %s\n", kernels().name);
}

#if defined(_WIN32)
std::vector<std::string> utf8_argv(int argc, char** argv) {
  int n = 0;
  LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n);
  std::vector<std::string> out;
  if (w == nullptr) {
    for (int i = 0; i < argc; ++i) out.emplace_back(argv[i]);
    return out;
  }
  for (int i = 0; i < n; ++i) {
    int len = WideCharToMultiByte(CP_UTF8, 0, w[i], -1, nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(std::max(len - 1, 0)), '\0');
    if (len > 1) WideCharToMultiByte(CP_UTF8, 0, w[i], -1, s.data(), len, nullptr, nullptr);
    out.push_back(std::move(s));
  }
  LocalFree(w);
  return out;
}

bool read_line(std::string& line) {
  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (!GetConsoleMode(in, &mode)) return static_cast<bool>(std::getline(std::cin, line));
  std::wstring w;
  wchar_t buf[512];
  for (;;) {
    DWORD got = 0;
    if (!ReadConsoleW(in, buf, 512, &got, nullptr) || got == 0) return !w.empty();
    w.append(buf, got);
    if (!w.empty() && w.back() == L'\n') break;
  }
  while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
  int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                nullptr, nullptr);
  line.assign(static_cast<size_t>(len), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), line.data(), len, nullptr,
                      nullptr);
  return true;
}
#else
std::vector<std::string> utf8_argv(int argc, char** argv) {
  std::vector<std::string> out;
  for (int i = 0; i < argc; ++i) out.emplace_back(argv[i]);
  return out;
}

bool read_line(std::string& line) { return static_cast<bool>(std::getline(std::cin, line)); }
#endif

void write_out(const std::string& s) {
  std::fwrite(s.data(), 1, s.size(), stdout);
  std::fflush(stdout);
}

int cmd_info(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  const ModelConfig& c = m->config();
  std::printf("model          %s\n", m->name().c_str());
  std::printf("architecture   %s\n", c.arch == Arch::Qwen35
                                         ? "qwen3.5 (gated deltanet + gated attention)"
                                         : "qwen3 (gqa attention)");
  std::printf("weights        %s\n", m->weight_label().c_str());
  std::printf("parameters     %.1f M\n", static_cast<double>(m->parameter_count()) / 1e6);
  std::printf("layers         %lld (%lld full attention, %lld linear attention)\n",
              static_cast<long long>(c.layers), static_cast<long long>(c.attn_layers()),
              static_cast<long long>(c.linear_layers()));
  std::printf("hidden         %lld, ffn %lld\n", static_cast<long long>(c.hidden),
              static_cast<long long>(c.intermediate));
  std::printf("attention      %lld heads, %lld kv heads, head dim %lld, rotary dim %lld\n",
              static_cast<long long>(c.heads), static_cast<long long>(c.kv_heads),
              static_cast<long long>(c.head_dim), static_cast<long long>(c.rotary_dim));
  if (c.linear_layers() > 0) {
    std::printf("deltanet       %lld key heads, %lld value heads, dims %lld/%lld, conv %lld\n",
                static_cast<long long>(c.lin_k_heads), static_cast<long long>(c.lin_v_heads),
                static_cast<long long>(c.lin_k_dim), static_cast<long long>(c.lin_v_dim),
                static_cast<long long>(c.conv_kernel));
  }
  std::printf("vocab          %lld (tokenizer %d)\n", static_cast<long long>(c.vocab),
              m->tokenizer().size());
  std::printf("mtp head       %s\n", m->weights().mtp.present ? "yes" : "no");
  std::printf("read per token %s\n", human_bytes(static_cast<double>(m->decode_bytes())).c_str());
  int64_t kv_per_token = c.attn_layers() * c.kv_heads * c.head_dim * 2 * 2;
  std::printf("kv per token   %s (f16)\n", human_bytes(static_cast<double>(kv_per_token)).c_str());
  if (c.linear_layers() > 0) {
    double state = static_cast<double>(
        c.linear_layers() *
        (c.lin_v_heads * c.lin_k_dim * c.lin_v_dim + (c.conv_kernel - 1) * c.conv_dim()) * 4);
    std::printf("fixed state    %s per sequence\n", human_bytes(state).c_str());
  }
  std::printf("cpu            %s [%s]\n", cpu_info().brand.c_str(), cpu_feature_string().c_str());
  std::printf("kernels        %s\n", kernels().name);
  std::printf("threads        %d\n", threads_of(a));
  return 0;
}

int cmd_convert(const Args& a) {
  std::string in = a.arg(0, "hugging face model directory");
  std::string out = a.arg(1, "output file");
  ConvertOptions o;
  if (!parse_wtype(a.get("type", "q8"), o.type)) fail("--type must be f32, f16, bf16, q8 or q4");
  o.threads = threads_of(a);
  double t0 = now_seconds();
  convert_model(in, out, o);
  SafeTensors check(out);
  std::fprintf(stderr, "wrote %s (%s, %zu tensors) in %.1fs\n", out.c_str(),
               human_bytes(static_cast<double>(check.file_size())).c_str(), check.tensors().size(),
               now_seconds() - t0);
  return 0;
}

std::string prompt_text(const Args& a) {
  if (a.has("prompt-file")) return read_text_file(a.get("prompt-file"));
  if (a.has("prompt")) return a.get("prompt");
  fail("give a prompt with -p or --prompt-file");
}

int cmd_run(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  check_draft(a, *m);
  ThreadPool pool(threads_of(a));
  pool.set_static(a.has("static"));
  Engine e(*m, pool, engine_options(a, 1));
  std::string text = prompt_text(a);
  if (a.has("chat") || a.has("system")) {
    std::vector<ChatMessage> msgs;
    if (a.has("system")) msgs.push_back({"system", a.get("system"), {}});
    msgs.push_back({"user", text, {}});
    text = render_chat(m->chat_format(), msgs, true, thinking_of(a));
  }
  GenerationRequest r;
  r.prompt = m->tokenizer().encode(text);
  r.max_tokens = a.get_int("max-tokens", 256);
  r.sampling = sampling_of(a, 0.0f);
  r.ignore_eos = a.has("ignore-eos");
  r.draft = draft_of(a);
  r.draft_tokens = static_cast<int>(a.get_int("draft-tokens", 0));
  if (!a.has("raw") && !a.has("chat") && !a.has("system")) write_out(text);
  GenerationResult res =
      e.generate(std::move(r), [](const std::string& piece) { write_out(piece); });
  write_out("\n");
  if (!res.error.empty()) std::fprintf(stderr, "error: %s\n", res.error.c_str());
  if (a.has("stats")) print_stats(res.stats);
  return res.reason == FinishReason::Error ? 1 : 0;
}

int cmd_chat(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  check_draft(a, *m);
  ThreadPool pool(threads_of(a));
  Engine e(*m, pool, engine_options(a, 1));
  std::vector<ChatMessage> history;
  if (a.has("system")) history.push_back({"system", a.get("system"), {}});
  int thinking = thinking_of(a);
  std::signal(SIGINT, on_sigint);
  std::fprintf(stderr,
               "chat with %s. /reset clears the history, /think toggles thinking, /exit quits.\n",
               m->name().c_str());
  for (;;) {
    write_out("\n> ");
    std::string line;
    if (!read_line(line)) break;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset") {
      history.erase(std::remove_if(history.begin(), history.end(),
                                   [](const ChatMessage& x) { return x.role != "system"; }),
                    history.end());
      std::fprintf(stderr, "history cleared\n");
      continue;
    }
    if (line == "/think") {
      thinking = thinking == 1 ? 0 : 1;
      std::fprintf(stderr, "thinking %s\n", thinking == 1 ? "on" : "off");
      continue;
    }
    if (line.empty()) continue;
    history.push_back({"user", line, {}});
    GenerationRequest r;
    r.prompt = m->tokenizer().encode(render_chat(m->chat_format(), history, true, thinking));
    r.max_tokens = a.get_int("max-tokens", 2048);
    r.sampling = sampling_of(a, 0.7f);
    if (!a.has("top-p")) r.sampling.top_p = 0.8f;
    if (!a.has("top-k")) r.sampling.top_k = 20;
    if (!a.has("seed")) r.sampling.seed = static_cast<uint64_t>(now_seconds() * 1000.0);
    r.draft = draft_of(a);
    r.draft_tokens = static_cast<int>(a.get_int("draft-tokens", 0));
    std::string reply;
    GenerationResult res;
    bool done = false;
    g_interrupted.store(false);
    int64_t id = e.submit(std::move(r), [&](const TokenEvent& ev) {
      reply += ev.text;
      write_out(ev.text);
      if (ev.finished) {
        done = true;
        res.stats = ev.stats;
        res.reason = ev.reason;
      }
    });
    while (!done) {
      if (g_interrupted.exchange(false)) e.cancel(id);
      e.step();
    }
    write_out("\n");
    history.push_back({"assistant", reply, {}});
    if (a.has("stats")) print_stats(res.stats);
  }
  return 0;
}

int cmd_serve(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  check_draft(a, *m);
  ThreadPool pool(threads_of(a));
  ServerOptions o;
  o.host = a.get("host", "127.0.0.1");
  o.port = static_cast<int>(a.get_int("port", 8080));
  o.engine = engine_options(a, a.get_int("parallel", 4));
  o.draft = draft_of(a);
  o.draft_tokens = static_cast<int>(a.get_int("draft-tokens", 0));
  o.model_name = a.get("name", m->name());
  Server server(*m, pool, o);
  int port = server.start();
  std::fprintf(
      stderr,
      "serving %s on http://%s:%d/v1 (%lld parallel sequences, %lld tokens of context each)\n",
      o.model_name.c_str(), o.host.c_str(), port, static_cast<long long>(o.engine.max_sequences),
      static_cast<long long>(a.get_int("ctx", 4096)));
  server.wait();
  return 0;
}

std::vector<int32_t> random_prompt(const Model& m, int64_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<int32_t> out;
  const int64_t limit = std::min<int64_t>(m.config().vocab, m.tokenizer().size());
  while (static_cast<int64_t>(out.size()) < n) {
    int32_t t = static_cast<int32_t>(rng.next() % static_cast<uint64_t>(limit));
    if (!m.tokenizer().is_added(t)) out.push_back(t);
  }
  return out;
}

void print_profile(const char* label, const std::vector<std::pair<std::string, double>>& phases,
                   double total, int64_t steps, const char* unit) {
  const double n = static_cast<double>(std::max<int64_t>(steps, 1));
  total = std::max(total, 1e-9);
  std::printf("  %s profile over %lld %s%s, %.2f ms per %s\n", label, static_cast<long long>(steps),
              unit, steps == 1 ? "" : "s", 1000.0 * total / n, unit);
  double accounted = 0.0;
  for (const auto& [name, secs] : phases) {
    accounted += secs;
    if (secs <= 0.0) continue;
    std::printf("    %-24s %8.3f ms %6.1f%%\n", name.c_str(), 1000.0 * secs / n,
                100.0 * secs / total);
  }
  const double rest = std::max(total - accounted, 0.0);
  std::printf("    %-24s %8.3f ms %6.1f%%\n", "scheduler and sampling", 1000.0 * rest / n,
              100.0 * rest / total);
}

int cmd_bench(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  check_draft(a, *m);
  const int64_t prompt_len = a.get_int("prompt", 512);
  const int64_t gen = a.get_int("gen", 128);
  const int64_t reps = a.get_int("reps", 3);
  const DraftMode draft = draft_of(a);
  const int draft_tokens = static_cast<int>(a.get_int("draft-tokens", 0));
  std::vector<int32_t> text_prompt;
  if (a.has("prompt-file")) {
    std::vector<ChatMessage> msgs = {{"user", read_text_file(a.get("prompt-file")), {}}};
    text_prompt = m->tokenizer().encode(render_chat(m->chat_format(), msgs, true, 0));
  }
  const double bytes = static_cast<double>(m->decode_bytes());
  std::printf("model %s, %s weights, %s read per decode step, kernels %s, cpu %s\n",
              m->name().c_str(), m->weight_label().c_str(), human_bytes(bytes).c_str(),
              kernels().name, cpu_info().brand.c_str());
  std::printf("%7s %5s %8s %14s %13s %15s %8s %9s\n", "threads", "batch", "prompt", "prefill tok/s",
              "decode tok/s", "aggregate tok/s", "GB/s", "accepted");
  for (int64_t threads : a.get_list("threads", default_thread_count())) {
    ThreadPool pool(static_cast<int>(threads));
    pool.set_static(a.has("static"));
    for (int64_t batch : a.get_list("batch", 1)) {
      EngineOptions eo = engine_options(a, batch);
      int64_t plen = text_prompt.empty() ? prompt_len : static_cast<int64_t>(text_prompt.size());
      eo.kv_tokens = batch * (plen + gen + 64);
      Engine e(*m, pool, eo);
      GenerationRequest warm;
      warm.prompt = random_prompt(*m, 32, 99);
      warm.max_tokens = 8;
      warm.ignore_eos = true;
      e.generate(std::move(warm));
      std::vector<double> prefill_v, decode_v, agg_v, bw_v, acc_v;
      for (int64_t rep = 0; rep < reps; ++rep) {
        std::vector<RequestStats> stats(static_cast<size_t>(batch));
        std::vector<double> first(static_cast<size_t>(batch), 0.0);
        std::vector<double> end(static_cast<size_t>(batch), 0.0);
        std::vector<int64_t> ids;
        std::map<int64_t, size_t> slot;
        for (int64_t b = 0; b < batch; ++b) {
          GenerationRequest r;
          r.prompt = text_prompt.empty()
                         ? random_prompt(*m, prompt_len, static_cast<uint64_t>(rep * 1000 + b))
                         : text_prompt;
          r.max_tokens = gen;
          r.ignore_eos = text_prompt.empty();
          r.draft = draft;
          r.draft_tokens = draft_tokens;
          size_t index = static_cast<size_t>(b);
          int64_t id = e.submit(std::move(r), [&, index](const TokenEvent& ev) {
            if (first[index] == 0.0 && !ev.tokens.empty()) first[index] = now_seconds();
            if (ev.finished) {
              end[index] = now_seconds();
              stats[index] = ev.stats;
            }
          });
          ids.push_back(id);
        }
        while (e.busy()) e.step();
        double sum_prefill = 0.0;
        double sum_decode = 0.0;
        int64_t tokens = 0;
        for (const RequestStats& s : stats) {
          sum_prefill += static_cast<double>(s.prompt_tokens) / std::max(s.prefill_seconds, 1e-9);
          sum_decode +=
              static_cast<double>(s.generated_tokens - 1) / std::max(s.decode_seconds, 1e-9);
          tokens += s.generated_tokens - 1;
        }
        double last_first = *std::max_element(first.begin(), first.end());
        double last_end = *std::max_element(end.begin(), end.end());
        prefill_v.push_back(sum_prefill / static_cast<double>(batch));
        decode_v.push_back(sum_decode / static_cast<double>(batch));
        agg_v.push_back(static_cast<double>(tokens) / std::max(last_end - last_first, 1e-9));
        const RequestStats& s0 = stats[0];
        bw_v.push_back(bytes * static_cast<double>(s0.decode_steps) /
                       std::max(s0.decode_seconds, 1e-9) / 1e9);
        acc_v.push_back(s0.drafted > 0
                            ? static_cast<double>(s0.accepted) / static_cast<double>(s0.drafted)
                            : -1.0);
      }
      auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      std::string accepted = acc_v[0] < 0.0 ? "-" : strf("%.0f%%", 100.0 * median(acc_v));
      std::printf(
          "%7lld %5lld %8lld %14.1f %13.1f %15.1f %8.1f %9s\n", static_cast<long long>(threads),
          static_cast<long long>(batch),
          static_cast<long long>(text_prompt.empty() ? prompt_len
                                                     : static_cast<int64_t>(text_prompt.size())),
          median(prefill_v), median(decode_v), median(agg_v), median(bw_v), accepted.c_str());
      if (a.has("profile")) {
        Runtime& rt = e.runtime();
        GenerationRequest r;
        r.prompt = text_prompt.empty() ? random_prompt(*m, prompt_len, 7) : text_prompt;
        r.max_tokens = gen;
        r.ignore_eos = text_prompt.empty();
        r.draft = draft;
        r.draft_tokens = draft_tokens;
        const int64_t prompt_tokens = static_cast<int64_t>(r.prompt.size());
        double t0 = 0.0;
        double t1 = 0.0;
        RequestStats st;
        std::vector<std::pair<std::string, double>> prefill;
        rt.set_profiling(true);
        rt.reset_profile();
        const double start = now_seconds();
        e.submit(std::move(r), [&](const TokenEvent& ev) {
          if (t0 == 0.0 && !ev.tokens.empty()) {
            prefill = rt.profile();
            rt.reset_profile();
            t0 = now_seconds();
          }
          if (ev.finished) {
            t1 = now_seconds();
            st = ev.stats;
          }
        });
        while (e.busy()) e.step();
        rt.set_profiling(false);
        print_profile("prefill", prefill, t0 - start, prompt_tokens, "token");
        print_profile("decode", rt.profile(), t1 - t0, st.decode_steps, "step");
      }
      std::fflush(stdout);
    }
  }
  return 0;
}

int cmd_perplexity(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  const int64_t ctx = a.get_int("ctx", 512);
  std::string text = read_text_file(a.get("file"));
  std::vector<int32_t> ids = m->tokenizer().encode(text, false);
  int64_t windows = static_cast<int64_t>(ids.size()) / ctx;
  if (a.has("chunks")) windows = std::min(windows, a.get_int("chunks", windows));
  if (windows < 1) fail("text is shorter than one window");
  ThreadPool pool(threads_of(a));
  RuntimeOptions ro;
  ro.max_batch_tokens = std::min<int64_t>(ctx, a.get_int("batch-tokens", 512));
  ro.kv_tokens = ctx + 16;
  ro.kv_f16 = a.get("kv", "f16") != "f32";
  Runtime rt(*m, pool, ro);
  const int64_t H = m->config().hidden;
  const int64_t V = m->config().vocab;
  const int64_t rows_per_logits = 64;
  std::vector<float> hidden(static_cast<size_t>(ro.max_batch_tokens * H));
  std::vector<float> logits(static_cast<size_t>(rows_per_logits * V));
  std::vector<double> row_nll(static_cast<size_t>(rows_per_logits));
  double nll = 0.0;
  int64_t scored = 0;
  double t0 = now_seconds();
  std::fprintf(stderr, "scoring %lld windows of %lld tokens (%zu tokens in file)\n",
               static_cast<long long>(windows), static_cast<long long>(ctx), ids.size());
  for (int64_t w = 0; w < windows; ++w) {
    Sequence* s = rt.acquire();
    const int32_t* toks = ids.data() + w * ctx;
    for (int64_t p = 0; p < ctx; p += ro.max_batch_tokens) {
      int64_t n = std::min(ro.max_batch_tokens, ctx - p);
      rt.reserve(*s, p + n, false);
      Batch b;
      b.add_run(s, toks + p, n, p, 0, true);
      rt.forward(b, hidden.data());
      for (int64_t r = 0; r < n; r += rows_per_logits) {
        int64_t k = std::min(rows_per_logits, n - r);
        rt.logits(hidden.data() + r * H, k, logits.data());
        pool.parallel_for(k, 1, [&](int64_t i0, int64_t i1) {
          for (int64_t i = i0; i < i1; ++i) {
            const int64_t pos = p + r + i;
            if (pos + 1 >= ctx) continue;
            const float* row = logits.data() + i * V;
            const float mx = *std::max_element(row, row + V);
            double z = 0.0;
            for (int64_t v = 0; v < V; ++v) z += std::exp(static_cast<double>(row[v] - mx));
            row_nll[static_cast<size_t>(i)] = std::log(z) + mx - row[toks[pos + 1]];
          }
        });
        for (int64_t i = 0; i < k; ++i) {
          if (p + r + i + 1 >= ctx) continue;
          nll += row_nll[static_cast<size_t>(i)];
          ++scored;
        }
      }
    }
    rt.release(s);
    if (!a.has("quiet")) {
      std::fprintf(stderr, "[%lld/%lld] ppl %.4f\n", static_cast<long long>(w + 1),
                   static_cast<long long>(windows), std::exp(nll / static_cast<double>(scored)));
    }
  }
  double secs = now_seconds() - t0;
  std::printf("perplexity %.4f over %lld tokens (%lld windows of %lld), %.1f tok/s\n",
              std::exp(nll / static_cast<double>(scored)), static_cast<long long>(scored),
              static_cast<long long>(windows), static_cast<long long>(ctx),
              static_cast<double>(windows * ctx) / secs);
  return 0;
}

int cmd_membw(const Args& a) {
  setup_runtime(a);
  const int64_t mb = a.get_int("mb", 1024);
  const int64_t n = mb * 1024 * 1024 / 8;
  Buffer<float> x(static_cast<size_t>(n));
  Buffer<float> y(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    x[static_cast<size_t>(i)] = 1.0f;
    y[static_cast<size_t>(i)] = 0.5f;
  }
  std::vector<int64_t> list;
  if (a.has("sweep")) {
    for (int t = 1; t <= default_thread_count(); ++t) list.push_back(t);
  } else {
    list = a.get_list("threads", default_thread_count());
  }
  std::printf("%7s %10s\n", "threads", "read GB/s");
  for (int64_t threads : list) {
    ThreadPool pool(static_cast<int>(threads));
    const int64_t block = 1 << 16;
    const int64_t blocks = n / block;
    std::vector<float> sink(static_cast<size_t>(blocks));
    double best = 0.0;
    for (int rep = 0; rep < 7; ++rep) {
      double t0 = now_seconds();
      pool.parallel_for(blocks, 4, [&](int64_t b0, int64_t b1) {
        for (int64_t b = b0; b < b1; ++b) {
          sink[static_cast<size_t>(b)] =
              kernels().dot(x.data() + b * block, y.data() + b * block, block);
        }
      });
      double secs = now_seconds() - t0;
      best = std::max(best, static_cast<double>(2 * blocks * block * 4) / secs / 1e9);
    }
    std::printf("%7lld %10.1f\n", static_cast<long long>(threads), best);
  }
  return 0;
}

std::unique_ptr<Tokenizer> load_tokenizer(const std::string& path) {
  if (path.size() >= 5 && path.substr(path.size() - 5) == ".json")
    return Tokenizer::from_json(read_text_file(path));
  if (is_directory(path))
    return Tokenizer::from_json(read_text_file(join_path(path, "tokenizer.json")));
  SafeTensors f(path);
  const TensorInfo* t = f.find("tokenizer.json");
  if (t == nullptr) fail(path + " has no tokenizer");
  return Tokenizer::from_json(
      std::string_view(reinterpret_cast<const char*>(t->data), static_cast<size_t>(t->nbytes)));
}

std::string to_hex(const std::string& s) {
  static const char* digits = "0123456789abcdef";
  std::string out;
  for (unsigned char c : s) {
    out += digits[c >> 4];
    out += digits[c & 15];
  }
  return out;
}

int cmd_tokenize(const Args& a) {
  auto tok = load_tokenizer(a.arg(0, "model or tokenizer path"));
  if (a.has("records")) {
    std::string data = read_text_file(a.get("records"));
    std::string mode = a.get("mode", "ids");
    ChatFormat format =
        a.get("format", "qwen35") == "qwen3" ? ChatFormat::Qwen3 : ChatFormat::Qwen35;
    std::string out;
    size_t pos = 0;
    while (pos < data.size()) {
      size_t nl = data.find('\n', pos);
      size_t len = static_cast<size_t>(std::stoull(data.substr(pos, nl - pos)));
      std::string_view text(data.data() + nl + 1, len);
      pos = nl + 1 + len;
      if (mode == "ids") {
        std::vector<int32_t> ids = tok->encode(text);
        for (size_t i = 0; i < ids.size(); ++i) {
          if (i) out += ' ';
          out += std::to_string(ids[i]);
        }
      } else if (mode == "nfc") {
        out += to_hex(nfc(text));
      } else if (mode == "roundtrip") {
        out += tok->decode(tok->encode(text)) == nfc(text) ? "same" : "diff";
      } else if (mode == "chat") {
        Json j = Json::parse(text);
        std::vector<ChatMessage> msgs;
        for (const Json& m : j["messages"].items()) {
          msgs.push_back(
              {m.get_string("role"), m.get_string("content"), m.get_string("reasoning_content")});
        }
        try {
          out += to_hex(render_chat(j.get_string("format") == "qwen3" ? ChatFormat::Qwen3 : format,
                                    msgs, j.get_bool("add", false),
                                    static_cast<int>(j.get_int("thinking", -1))));
        } catch (const std::exception&) {
          out += "ERROR";
        }
      } else {
        fail("unknown --mode " + mode);
      }
      out += '\n';
    }
    write_out(out);
    return 0;
  }
  std::string text = a.has("file") ? read_text_file(a.get("file")) : a.get("prompt");
  std::vector<int32_t> ids = tok->encode(text);
  std::string out;
  if (a.has("ids")) {
    for (size_t i = 0; i < ids.size(); ++i) out += (i ? " " : "") + std::to_string(ids[i]);
  } else {
    for (int32_t id : ids) {
      std::string piece = tok->token_bytes(id);
      out += std::to_string(id) + "\t" + Json(utf8_sanitize(piece)).dump() + "\n";
    }
  }
  write_out(out + (a.has("ids") ? "\n" : ""));
  std::fprintf(stderr, "%zu tokens\n", ids.size());
  return 0;
}

int cmd_logits(const Args& a) {
  setup_runtime(a);
  auto m = load_model(a, 0);
  std::vector<int32_t> ids;
  {
    std::ifstream in(a.get("ids-file"));
    int32_t v;
    while (in >> v) ids.push_back(v);
  }
  if (ids.empty()) fail("--ids-file is empty");
  ThreadPool pool(threads_of(a));
  RuntimeOptions ro;
  ro.max_batch_tokens = std::min<int64_t>(static_cast<int64_t>(ids.size()), 256);
  ro.kv_tokens = static_cast<int64_t>(ids.size()) + 16;
  ro.kv_f16 = a.get("kv", "f16") != "f32";
  const int64_t H = m->config().hidden;
  const int64_t V = m->config().vocab;
  if (a.has("out")) {
    Runtime rt(*m, pool, ro);
    Sequence* s = rt.acquire();
    std::ofstream out(a.get("out"), std::ios::binary);
    std::vector<float> hidden(static_cast<size_t>(ro.max_batch_tokens * H));
    std::vector<float> logits(static_cast<size_t>(16 * V));
    for (size_t p = 0; p < ids.size(); p += static_cast<size_t>(ro.max_batch_tokens)) {
      int64_t n = std::min<int64_t>(ro.max_batch_tokens, static_cast<int64_t>(ids.size() - p));
      rt.reserve(*s, static_cast<int64_t>(p) + n, false);
      Batch b;
      b.add_run(s, ids.data() + p, n, static_cast<int64_t>(p), 0, true);
      rt.forward(b, hidden.data());
      for (int64_t r = 0; r < n; r += 16) {
        int64_t k = std::min<int64_t>(16, n - r);
        rt.logits(hidden.data() + r * H, k, logits.data());
        out.write(reinterpret_cast<const char*>(logits.data()),
                  static_cast<std::streamsize>(k * V * 4));
      }
    }
  }
  if (a.has("greedy")) {
    EngineOptions eo;
    eo.kv_tokens = static_cast<int64_t>(ids.size()) + a.get_int("greedy", 32) + 64;
    eo.kv_f16 = ro.kv_f16;
    Engine e(*m, pool, eo);
    GenerationRequest r;
    r.prompt = ids;
    r.max_tokens = a.get_int("greedy", 32);
    r.ignore_eos = true;
    r.draft = draft_of(a);
    r.draft_tokens = static_cast<int>(a.get_int("draft-tokens", 0));
    GenerationResult res = e.generate(std::move(r));
    std::string line;
    for (size_t i = 0; i < res.tokens.size(); ++i)
      line += (i ? " " : "") + std::to_string(res.tokens[i]);
    write_out(line + "\n");
  }
  return 0;
}

void usage() {
  std::fprintf(
      stderr,
      "infer: CPU inference engine for Qwen3.5 and Qwen3 models\n\n"
      "usage: infer <command> [options]\n\n"
      "  info <model>                         model, memory and cpu summary\n"
      "  convert <hf-dir> <out.infer>         convert weights (--type q8|q4|bf16)\n"
      "  run <model> -p <text>                generate text (--chat, --draft mtp|ngram, --stats)\n"
      "  chat <model>                         interactive chat\n"
      "  serve <model>                        openai compatible server (--port, --parallel)\n"
      "  bench <model>                        prefill and decode speed (--threads, --batch, "
      "--draft, --profile)\n"
      "  perplexity <model> --file <txt>      perplexity over fixed windows (--ctx)\n"
      "  membw                                memory read bandwidth (--sweep)\n"
      "  tokenize <model> -p <text>           show tokens\n"
      "  logits <model> --ids-file <f>        dump logits for verification\n\n"
      "common options: --threads N, --ctx N, --kv f16|f32, --kernels scalar|avx2|vnni|avx512\n"
      "<model> is a Hugging Face model directory or a .infer file from convert\n");
}

}

int main(int argc, char** argv) {
#if defined(_WIN32)
  SetConsoleOutputCP(CP_UTF8);
#endif
  std::vector<std::string> args = utf8_argv(argc, argv);
  if (args.size() < 2 || args[1] == "-h" || args[1] == "--help" || args[1] == "help") {
    usage();
    return args.size() < 2 ? 1 : 0;
  }
  const std::string cmd = args[1];
  try {
    Args a(args, 2, kFlags);
    if (cmd == "info") return cmd_info(a);
    if (cmd == "convert") return cmd_convert(a);
    if (cmd == "run") return cmd_run(a);
    if (cmd == "chat") return cmd_chat(a);
    if (cmd == "serve") return cmd_serve(a);
    if (cmd == "bench") return cmd_bench(a);
    if (cmd == "perplexity") return cmd_perplexity(a);
    if (cmd == "membw") return cmd_membw(a);
    if (cmd == "tokenize") return cmd_tokenize(a);
    if (cmd == "logits") return cmd_logits(a);
    usage();
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

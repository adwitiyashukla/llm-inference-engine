#include "server.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <deque>
#include <map>

#include "json.h"
#include "unicode.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace infer {

namespace {

#if defined(_WIN32)
using SocketHandle = SOCKET;
const SocketHandle kInvalidSocket = INVALID_SOCKET;
void close_socket(SocketHandle s) { closesocket(s); }
struct NetInit {
  NetInit() {
    WSADATA data;
    WSAStartup(MAKEWORD(2, 2), &data);
  }
  ~NetInit() { WSACleanup(); }
};
#else
using SocketHandle = int;
const SocketHandle kInvalidSocket = -1;
void close_socket(SocketHandle s) { ::close(s); }
struct NetInit {};
#endif

NetInit& net_init() {
  static NetInit init;
  return init;
}

bool send_all(SocketHandle s, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
#if defined(MSG_NOSIGNAL)
    const int flags = MSG_NOSIGNAL;
#else
    const int flags = 0;
#endif
    auto n = ::send(s, data.data() + sent,
                    static_cast<int>(std::min<size_t>(data.size() - sent, 1 << 20)), flags);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
};

bool read_request(SocketHandle s, HttpRequest& req) {
  std::string buf;
  char chunk[8192];
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    auto n = ::recv(s, chunk, sizeof(chunk), 0);
    if (n <= 0) return false;
    buf.append(chunk, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
    if (header_end == std::string::npos && buf.size() > 65536) return false;
  }
  std::string head = buf.substr(0, header_end);
  size_t line_end = head.find("\r\n");
  std::string first = head.substr(0, line_end);
  size_t sp1 = first.find(' ');
  size_t sp2 = first.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  req.method = first.substr(0, sp1);
  req.path = first.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t q = req.path.find('?');
  if (q != std::string::npos) req.path = req.path.substr(0, q);
  size_t pos = line_end == std::string::npos ? head.size() : line_end + 2;
  while (pos < head.size()) {
    size_t end = head.find("\r\n", pos);
    if (end == std::string::npos) end = head.size();
    std::string line = head.substr(pos, end - pos);
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = line.substr(0, colon);
      std::transform(key.begin(), key.end(), key.begin(),
                     [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      size_t v = line.find_first_not_of(' ', colon + 1);
      req.headers[key] = v == std::string::npos ? std::string() : line.substr(v);
    }
    pos = end + 2;
  }
  size_t length = 0;
  auto it = req.headers.find("content-length");
  if (it != req.headers.end())
    length = static_cast<size_t>(std::strtoull(it->second.c_str(), nullptr, 10));
  if (length > (size_t{32} << 20)) return false;
  req.body = buf.substr(header_end + 4);
  while (req.body.size() < length) {
    auto n = ::recv(s, chunk, sizeof(chunk), 0);
    if (n <= 0) return false;
    req.body.append(chunk, static_cast<size_t>(n));
  }
  req.body.resize(length);
  return true;
}

const char* status_text(int code) {
  switch (code) {
    case 200:
      return "OK";
    case 204:
      return "No Content";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 503:
      return "Service Unavailable";
    default:
      return "Internal Server Error";
  }
}

std::string response(int code, const std::string& body, const char* type = "application/json") {
  std::string out = strf("HTTP/1.1 %d %s\r\n", code, status_text(code));
  out += std::string("Content-Type: ") + type + "\r\n";
  out += strf("Content-Length: %zu\r\n", body.size());
  out += "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n";
  out += "Connection: close\r\n\r\n";
  out += body;
  return out;
}

std::string error_body(const std::string& message) {
  Json err = Json::object();
  err.set("message", Json(message));
  err.set("type", Json("invalid_request_error"));
  Json doc = Json::object();
  doc.set("error", std::move(err));
  return doc.dump();
}

struct EventQueue {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<TokenEvent> events;
  void push(const TokenEvent& ev) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      events.push_back(ev);
    }
    cv.notify_one();
  }
  TokenEvent pop() {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !events.empty(); });
    TokenEvent ev = std::move(events.front());
    events.pop_front();
    return ev;
  }
};

class ReasoningSplitter {
 public:
  enum class State { Detect, Reasoning, Content };
  explicit ReasoningSplitter(State start)
      : state_(start), started_content_(start == State::Content) {}

  void push(const std::string& text, std::string& reasoning, std::string& content) {
    buf_ += text;
    process(reasoning, content, false);
  }
  void flush(std::string& reasoning, std::string& content) { process(reasoning, content, true); }

 private:
  void process(std::string& reasoning, std::string& content, bool final) {
    for (;;) {
      if (state_ == State::Detect) {
        size_t start = buf_.find_first_not_of("\n ");
        if (start == std::string::npos) {
          if (final) state_ = State::Content;
          if (!final) return;
          continue;
        }
        std::string rest = buf_.substr(start);
        const std::string open = "<think>";
        if (rest.size() >= open.size() && rest.compare(0, open.size(), open) == 0) {
          buf_ = rest.substr(open.size());
          state_ = State::Reasoning;
          continue;
        }
        if (!final && open.compare(0, rest.size(), rest) == 0) return;
        state_ = State::Content;
        continue;
      }
      if (state_ == State::Content) {
        if (!started_content_) {
          size_t k = buf_.find_first_not_of('\n');
          if (k == std::string::npos) {
            if (final) buf_.clear();
            return;
          }
          buf_ = buf_.substr(k);
          started_content_ = true;
        }
        content += buf_;
        buf_.clear();
        return;
      }
      const std::string close = "</think>";
      size_t p = buf_.find(close);
      if (p != std::string::npos) {
        reasoning += strip_leading(buf_.substr(0, p));
        buf_ = buf_.substr(p + close.size());
        state_ = State::Content;
        continue;
      }
      size_t hold = 0;
      for (size_t len = std::min(buf_.size(), close.size() - 1); len > 0; --len) {
        if (buf_.compare(buf_.size() - len, len, close, 0, len) == 0) {
          hold = len;
          break;
        }
      }
      if (final) hold = 0;
      reasoning += strip_leading(buf_.substr(0, buf_.size() - hold));
      buf_.erase(0, buf_.size() - hold);
      return;
    }
  }
  std::string strip_leading(const std::string& s) {
    if (started_reasoning_) return s;
    size_t k = s.find_first_not_of('\n');
    if (k == std::string::npos) return {};
    started_reasoning_ = true;
    return s.substr(k);
  }

  State state_;
  std::string buf_;
  bool started_reasoning_ = false;
  bool started_content_ = false;
};

std::string message_text(const Json& content) {
  if (content.is_string()) return content.as_string();
  if (content.is_null()) return {};
  if (!content.is_array()) fail("message content must be a string or an array of parts");
  std::string out;
  for (const Json& part : content.items()) {
    if (part.get_string("type", "text") != "text") fail("only text content parts are supported");
    out += part.get_string("text");
  }
  return out;
}

}

Server::Server(const Model& model, ThreadPool& pool, const ServerOptions& options)
    : model_(model), opt_(options) {
  net_init();
  engine_ = std::make_unique<Engine>(model, pool, opt_.engine);
  if (opt_.model_name.empty()) opt_.model_name = model.name();
}

Server::~Server() { stop(); }

int Server::start() {
  SocketHandle s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (s == kInvalidSocket) fail("cannot create socket");
  int yes = 1;
  ::setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(opt_.port));
  if (::inet_pton(AF_INET, opt_.host.c_str(), &addr.sin_addr) != 1) {
    close_socket(s);
    fail("invalid host address " + opt_.host);
  }
  if (::bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close_socket(s);
    fail(strf("cannot bind %s:%d", opt_.host.c_str(), opt_.port));
  }
  if (::listen(s, 64) != 0) {
    close_socket(s);
    fail("cannot listen");
  }
  sockaddr_in bound{};
  socklen_t len = sizeof(bound);
  ::getsockname(s, reinterpret_cast<sockaddr*>(&bound), &len);
  listener_ = static_cast<intptr_t>(s);
  running_ = true;
  engine_thread_ = std::thread([this] { engine_loop(); });
  accept_thread_ = std::thread([this] { accept_loop(); });
  return ntohs(bound.sin_port);
}

void Server::stop() {
  if (!running_.exchange(false)) return;
  if (listener_ != -1) {
    SocketHandle s = static_cast<SocketHandle>(listener_);
#if defined(_WIN32)
    ::shutdown(s, SD_BOTH);
#else
    ::shutdown(s, SHUT_RDWR);
#endif
    close_socket(s);
  }
  notify();
  if (accept_thread_.joinable()) accept_thread_.join();
  listener_ = -1;
  for (int i = 0; i < 500 && connections_.load() > 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (engine_thread_.joinable()) engine_thread_.join();
}

void Server::wait() {
  if (accept_thread_.joinable()) accept_thread_.join();
}

void Server::notify() {
  { std::lock_guard<std::mutex> lock(mutex_); }
  cv_.notify_all();
}

void Server::engine_loop() {
  for (;;) {
    bool worked = false;
    try {
      worked = engine_->step();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "engine error: %s\n", e.what());
    }
    if (!running_.load() && !engine_->busy()) return;
    if (!worked) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait_for(lock, std::chrono::milliseconds(20),
                   [&] { return !running_.load() || engine_->busy(); });
    }
  }
}

void Server::accept_loop() {
  while (running_.load()) {
    SocketHandle c = ::accept(static_cast<SocketHandle>(listener_), nullptr, nullptr);
    if (c == kInvalidSocket) {
      if (!running_.load()) break;
      continue;
    }
    int yes = 1;
    ::setsockopt(c, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&yes), sizeof(yes));
#if defined(_WIN32)
    DWORD timeout_ms = 60000;
    ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&timeout_ms),
                 sizeof(timeout_ms));
#else
    timeval tv{};
    tv.tv_sec = 60;
    ::setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
#if defined(__APPLE__)
    ::setsockopt(c, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof(yes));
#endif
    connections_.fetch_add(1);
    std::thread([this, c] {
      handle(static_cast<intptr_t>(c));
      close_socket(c);
      connections_.fetch_sub(1);
    }).detach();
  }
}

void Server::handle(intptr_t client) {
  SocketHandle s = static_cast<SocketHandle>(client);
  HttpRequest req;
  if (!read_request(s, req)) return;
  const double t0 = now_seconds();
  if (req.method == "OPTIONS") {
    send_all(s, response(204, ""));
    return;
  }
  if (req.method == "GET" && (req.path == "/health" || req.path == "/")) {
    send_all(s, response(200, R"({"status":"ok"})"));
    return;
  }
  if (req.method == "GET" && req.path == "/v1/models") {
    Json m = Json::object();
    m.set("id", Json(opt_.model_name));
    m.set("object", Json("model"));
    m.set("owned_by", Json("local"));
    Json data = Json::array();
    data.push(std::move(m));
    Json doc = Json::object();
    doc.set("object", Json("list"));
    doc.set("data", std::move(data));
    send_all(s, response(200, doc.dump()));
    return;
  }
  const bool chat = req.path == "/v1/chat/completions";
  const bool completion = req.path == "/v1/completions";
  if (!chat && !completion) {
    send_all(s, response(404, error_body("unknown endpoint " + req.path)));
    return;
  }
  if (req.method != "POST") {
    send_all(s, response(405, error_body("use POST")));
    return;
  }
  GenerationRequest gen;
  bool stream = false;
  bool include_usage = false;
  ReasoningSplitter::State start_state = ReasoningSplitter::State::Content;
  try {
    Json body = Json::parse(req.body);
    stream = body.get_bool("stream", false);
    include_usage = body["stream_options"].get_bool("include_usage", false);
    std::string prompt;
    if (chat) {
      std::vector<ChatMessage> messages;
      for (const Json& m : body["messages"].items()) {
        ChatMessage msg;
        msg.role = m.get_string("role");
        msg.content = message_text(m["content"]);
        msg.reasoning = m.get_string("reasoning_content");
        messages.push_back(std::move(msg));
      }
      if (messages.empty()) fail("messages must not be empty");
      int thinking = -1;
      const Json& kwargs = body["chat_template_kwargs"];
      if (kwargs.contains("enable_thinking"))
        thinking = kwargs.get_bool("enable_thinking", false) ? 1 : 0;
      if (body.contains("enable_thinking"))
        thinking = body.get_bool("enable_thinking", false) ? 1 : 0;
      prompt = render_chat(model_.chat_format(), messages, true, thinking);
      if (model_.chat_format() == ChatFormat::Qwen35) {
        start_state =
            thinking == 1 ? ReasoningSplitter::State::Reasoning : ReasoningSplitter::State::Content;
      } else {
        start_state =
            thinking == 0 ? ReasoningSplitter::State::Content : ReasoningSplitter::State::Detect;
      }
    } else {
      prompt = body.get_string("prompt");
    }
    gen.prompt = model_.tokenizer().encode(prompt);
    gen.max_tokens = body.get_int("max_completion_tokens", body.get_int("max_tokens", 1024));
    if (gen.max_tokens < 1) fail("max_tokens must be positive");
    gen.sampling.temperature = static_cast<float>(body.get_number("temperature", 0.7));
    gen.sampling.top_p = static_cast<float>(body.get_number("top_p", 0.8));
    gen.sampling.top_k = static_cast<int>(body.get_int("top_k", 20));
    gen.sampling.min_p = static_cast<float>(body.get_number("min_p", 0.0));
    gen.sampling.presence_penalty = static_cast<float>(body.get_number("presence_penalty", 0.0));
    gen.sampling.frequency_penalty = static_cast<float>(body.get_number("frequency_penalty", 0.0));
    gen.sampling.repetition_penalty =
        static_cast<float>(body.get_number("repetition_penalty", 1.0));
    gen.sampling.seed = static_cast<uint64_t>(body.get_int("seed", counter_.load() + 1));
    const Json& stop = body["stop"];
    if (stop.is_string()) gen.stop.push_back(stop.as_string());
    for (const Json& s2 : stop.items()) gen.stop.push_back(s2.as_string());
    gen.draft = opt_.draft;
    gen.draft_tokens = opt_.draft_tokens;
    if (body.contains("draft")) {
      if (!parse_draft_mode(body.get_string("draft"), gen.draft))
        fail("draft must be none, ngram or mtp");
    }
    gen.draft_tokens = static_cast<int>(body.get_int("draft_tokens", gen.draft_tokens));
  } catch (const std::exception& e) {
    send_all(s, response(400, error_body(e.what())));
    return;
  }

  const int64_t prompt_tokens = static_cast<int64_t>(gen.prompt.size());
  const int64_t number = counter_.fetch_add(1) + 1;
  const std::string id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(number);
  const int64_t created = static_cast<int64_t>(std::time(nullptr));
  auto queue = std::make_shared<EventQueue>();
  int64_t handle_id =
      engine_->submit(std::move(gen), [queue](const TokenEvent& ev) { queue->push(ev); });
  notify();

  ReasoningSplitter splitter(chat ? start_state : ReasoningSplitter::State::Content);
  auto chunk = [&](const std::string& reasoning, const std::string& content, const char* finish,
                   bool role) {
    Json choice = Json::object();
    choice.set("index", Json(0));
    if (chat) {
      Json delta = Json::object();
      if (role) delta.set("role", Json("assistant"));
      if (!reasoning.empty()) delta.set("reasoning_content", Json(reasoning));
      if (!content.empty() || role) delta.set("content", Json(content));
      choice.set("delta", std::move(delta));
    } else {
      choice.set("text", Json(content));
    }
    choice.set("finish_reason", finish != nullptr ? Json(finish) : Json());
    Json doc = Json::object();
    doc.set("id", Json(id));
    doc.set("object", Json(chat ? "chat.completion.chunk" : "text_completion"));
    doc.set("created", Json(created));
    doc.set("model", Json(opt_.model_name));
    Json choices = Json::array();
    choices.push(std::move(choice));
    doc.set("choices", std::move(choices));
    return "data: " + doc.dump() + "\n\n";
  };
  auto usage = [&](int64_t completion_tokens) {
    Json u = Json::object();
    u.set("prompt_tokens", Json(prompt_tokens));
    u.set("completion_tokens", Json(completion_tokens));
    u.set("total_tokens", Json(prompt_tokens + completion_tokens));
    return u;
  };

  bool alive = true;
  if (stream) {
    std::string head =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
        "Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    alive = send_all(s, head) && send_all(s, chunk("", "", nullptr, chat));
  }
  std::string reasoning_all;
  std::string content_all;
  TokenEvent last;
  for (;;) {
    TokenEvent ev = queue->pop();
    std::string r;
    std::string c;
    splitter.push(ev.text, r, c);
    if (ev.finished) splitter.flush(r, c);
    reasoning_all += r;
    content_all += c;
    if (stream && alive && (!r.empty() || !c.empty())) {
      alive = send_all(s, chunk(r, c, nullptr, false));
      if (!alive) engine_->cancel(handle_id);
    }
    if (ev.finished) {
      last = ev;
      break;
    }
  }
  const char* finish = last.reason == FinishReason::Length ? "length" : "stop";
  const int64_t generated = last.stats.generated_tokens;
  if (stream) {
    if (alive) {
      std::string tail = chunk("", "", finish, false);
      if (include_usage) {
        Json doc = Json::object();
        doc.set("id", Json(id));
        doc.set("object", Json(chat ? "chat.completion.chunk" : "text_completion"));
        doc.set("created", Json(created));
        doc.set("model", Json(opt_.model_name));
        doc.set("choices", Json::array());
        doc.set("usage", usage(generated));
        tail += "data: " + doc.dump() + "\n\n";
      }
      tail += "data: [DONE]\n\n";
      send_all(s, tail);
    }
  } else if (last.reason == FinishReason::Error) {
    send_all(s, response(500, error_body(last.error)));
  } else {
    Json choice = Json::object();
    choice.set("index", Json(0));
    if (chat) {
      Json message = Json::object();
      message.set("role", Json("assistant"));
      message.set("content", Json(content_all));
      if (!reasoning_all.empty()) message.set("reasoning_content", Json(reasoning_all));
      choice.set("message", std::move(message));
    } else {
      choice.set("text", Json(content_all));
    }
    choice.set("finish_reason", Json(finish));
    Json choices = Json::array();
    choices.push(std::move(choice));
    Json doc = Json::object();
    doc.set("id", Json(id));
    doc.set("object", Json(chat ? "chat.completion" : "text_completion"));
    doc.set("created", Json(created));
    doc.set("model", Json(opt_.model_name));
    doc.set("choices", std::move(choices));
    doc.set("usage", usage(generated));
    send_all(s, response(200, doc.dump()));
  }
  if (opt_.log) {
    const RequestStats& st = last.stats;
    double tps = st.decode_seconds > 0.0
                     ? static_cast<double>(std::max<int64_t>(generated - 1, 0)) / st.decode_seconds
                     : 0.0;
    std::fprintf(stderr,
                 "%s %s prompt=%lld completion=%lld prefill=%.2fs decode=%.1f tok/s drafted=%lld "
                 "accepted=%lld total=%.2fs %s\n",
                 req.method.c_str(), req.path.c_str(), static_cast<long long>(prompt_tokens),
                 static_cast<long long>(generated), st.prefill_seconds, tps,
                 static_cast<long long>(st.drafted), static_cast<long long>(st.accepted),
                 now_seconds() - t0, finish_reason_name(last.reason));
  }
}

}

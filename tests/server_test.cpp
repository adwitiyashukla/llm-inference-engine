#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "engine.h"
#include "json.h"
#include "server.h"
#include "test_models.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace infer;

namespace {

#if defined(_WIN32)
using Sock = SOCKET;
void close_sock(Sock s) { closesocket(s); }
#else
using Sock = int;
void close_sock(Sock s) { ::close(s); }
#endif

int g_port = 0;
std::unique_ptr<Model> g_model;

std::string http(const std::string& method, const std::string& path, const std::string& body) {
  Sock s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(g_port));
  ::inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
  if (::connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close_sock(s);
    return {};
  }
  std::string req =
      method + " " + path + " HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n";
  req += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
  ::send(s, req.data(), static_cast<int>(req.size()), 0);
  std::string out;
  char buf[4096];
  for (;;) {
    auto n = ::recv(s, buf, sizeof(buf), 0);
    if (n <= 0) break;
    out.append(buf, static_cast<size_t>(n));
  }
  close_sock(s);
  return out;
}

int status_of(const std::string& resp) {
  return resp.size() > 12 ? std::atoi(resp.substr(9, 3).c_str()) : 0;
}

std::string body_of(const std::string& resp) {
  size_t p = resp.find("\r\n\r\n");
  return p == std::string::npos ? std::string() : resp.substr(p + 4);
}

std::string chat_body(const std::string& user, bool stream, int max_tokens = 24) {
  Json msg = Json::object();
  msg.set("role", Json("user"));
  msg.set("content", Json(user));
  Json messages = Json::array();
  messages.push(std::move(msg));
  Json doc = Json::object();
  doc.set("model", Json("tiny"));
  doc.set("messages", std::move(messages));
  doc.set("max_tokens", Json(max_tokens));
  doc.set("temperature", Json(0.0));
  doc.set("stream", Json(stream));
  if (stream) {
    Json opts = Json::object();
    opts.set("include_usage", Json(true));
    doc.set("stream_options", std::move(opts));
  }
  return doc.dump();
}

std::string expected_reply(const std::string& user, int max_tokens = 24) {
  ThreadPool pool(2);
  EngineOptions o;
  Engine e(*g_model, pool, o);
  GenerationRequest r;
  r.prompt = g_model->tokenizer().encode(
      render_chat(g_model->chat_format(), {{"user", user, {}}}, true, -1));
  r.max_tokens = max_tokens;
  return e.generate(std::move(r)).text;
}

std::string content_of(const std::string& resp) {
  Json doc = Json::parse(body_of(resp));
  return doc["choices"].at(0)["message"].get_string("content");
}

}

TEST(health_and_models) {
  std::string h = http("GET", "/health", "");
  EXPECT(status_of(h) == 200);
  EXPECT(body_of(h).find("ok") != std::string::npos);
  std::string m = http("GET", "/v1/models", "");
  EXPECT(status_of(m) == 200);
  EXPECT(Json::parse(body_of(m))["data"].at(0).get_string("id") == "tiny");
}

TEST(chat_completion_matches_the_engine) {
  std::string resp = http("POST", "/v1/chat/completions", chat_body("hello there", false));
  EXPECT(status_of(resp) == 200);
  Json doc = Json::parse(body_of(resp));
  EXPECT(doc.get_string("object") == "chat.completion");
  EXPECT(doc["usage"].get_int("completion_tokens", 0) == 24);
  EXPECT(content_of(resp) == expected_reply("hello there"));
}

TEST(streaming_matches_non_streaming) {
  std::string resp = http("POST", "/v1/chat/completions", chat_body("stream this", true));
  EXPECT(status_of(resp) == 200);
  std::string body = body_of(resp);
  std::string text;
  bool done = false;
  bool finish = false;
  bool usage = false;
  size_t pos = 0;
  while ((pos = body.find("data: ", pos)) != std::string::npos) {
    size_t end = body.find("\n\n", pos);
    std::string payload = body.substr(pos + 6, end - pos - 6);
    pos = end;
    if (payload == "[DONE]") {
      done = true;
      continue;
    }
    Json ev = Json::parse(payload);
    if (ev.contains("usage")) usage = true;
    if (ev["choices"].size() == 0) continue;
    const Json& c = ev["choices"].at(0);
    text += c["delta"].get_string("content");
    if (c["finish_reason"].is_string()) finish = true;
  }
  EXPECT(done);
  EXPECT(finish);
  EXPECT(usage);
  EXPECT(text == expected_reply("stream this"));
}

TEST(concurrent_requests_match_solo_runs) {
  std::vector<std::string> prompts = {"one", "two two", "three three three", "four", "five five"};
  std::vector<std::string> got(prompts.size());
  std::vector<std::thread> threads;
  for (size_t i = 0; i < prompts.size(); ++i) {
    threads.emplace_back([&, i] {
      got[i] = content_of(http("POST", "/v1/chat/completions", chat_body(prompts[i], false)));
    });
  }
  for (auto& t : threads) t.join();
  for (size_t i = 0; i < prompts.size(); ++i) EXPECT(got[i] == expected_reply(prompts[i]));
}

TEST(text_completion_endpoint) {
  std::string resp =
      http("POST", "/v1/completions", R"({"prompt":"the and","max_tokens":5,"temperature":0})");
  EXPECT(status_of(resp) == 200);
  Json doc = Json::parse(body_of(resp));
  EXPECT(doc["choices"].at(0)["text"].is_string());
  EXPECT(doc["usage"].get_int("completion_tokens", 0) == 5);
}

TEST(bad_requests_get_errors) {
  EXPECT(status_of(http("POST", "/v1/chat/completions", "{not json")) == 400);
  EXPECT(status_of(http("POST", "/v1/chat/completions", R"({"messages":[]})")) == 400);
  EXPECT(status_of(http("GET", "/v1/nothing", "")) == 404);
}

int main() {
  g_model = make_model(hybrid_config(), WType::Q8, true);
  ThreadPool pool(2);
  ServerOptions o;
  o.port = 0;
  o.model_name = "tiny";
  o.engine.max_sequences = 3;
  o.engine.kv_tokens = 4096;
  o.log = false;
  Server server(*g_model, pool, o);
  g_port = server.start();
  int rc = check::run_all();
  server.stop();
  return rc;
}

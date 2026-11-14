#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "engine.h"

namespace infer {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  EngineOptions engine;
  DraftMode draft = DraftMode::None;
  int draft_tokens = 0;
  std::string model_name;
  bool log = true;
};

class Server {
 public:
  Server(const Model& model, ThreadPool& pool, const ServerOptions& options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  int start();
  void stop();
  void wait();
  Engine& engine() { return *engine_; }

 private:
  void accept_loop();
  void engine_loop();
  void handle(intptr_t client);
  void notify();

  const Model& model_;
  ServerOptions opt_;
  std::unique_ptr<Engine> engine_;
  intptr_t listener_ = -1;
  std::atomic<bool> running_{false};
  std::thread accept_thread_;
  std::thread engine_thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<int> connections_{0};
  std::atomic<int64_t> counter_{0};
};

}

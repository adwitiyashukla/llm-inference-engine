#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace infer {

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  std::string path_;
#if defined(_WIN32)
  void* file_ = nullptr;
  void* mapping_ = nullptr;
#endif
};

enum class DType : uint8_t { Bool, U8, I8, U16, I16, F16, BF16, U32, I32, F32, U64, I64, F64 };

size_t dtype_size(DType type);
const char* dtype_name(DType type);
bool parse_dtype(std::string_view name, DType& out);

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  uint64_t nbytes = 0;
  int64_t numel() const;
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);

  const std::vector<TensorInfo>& tensors() const { return tensors_; }
  const TensorInfo* find(std::string_view name) const;
  bool has_metadata(std::string_view key) const;
  std::string metadata(std::string_view key) const;
  const std::string& path() const { return file_->path(); }
  size_t file_size() const { return file_->size(); }

 private:
  std::unique_ptr<MappedFile> file_;
  std::vector<TensorInfo> tensors_;
  std::unordered_map<std::string, size_t> index_;
  std::vector<std::pair<std::string, std::string>> metadata_;
};

struct TensorSpec {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  std::function<void(uint8_t* dst, uint64_t nbytes)> fill;
};

void write_safetensors(const std::string& path,
                       const std::vector<std::pair<std::string, std::string>>& metadata,
                       const std::vector<TensorSpec>& tensors, size_t alignment = 64);

}

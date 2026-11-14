#include "safetensors.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>

#include "common.h"
#include "json.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace infer {

static std::filesystem::path fs_path(const std::string& utf8) {
  return std::filesystem::path(
      std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

MappedFile::MappedFile(const std::string& path) : path_(path) {
#if defined(_WIN32)
  std::wstring wide = fs_path(path).wstring();
  HANDLE file = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) fail("cannot open " + path);
  LARGE_INTEGER size;
  if (!GetFileSizeEx(file, &size)) {
    CloseHandle(file);
    fail("cannot stat " + path);
  }
  size_ = static_cast<size_t>(size.QuadPart);
  file_ = file;
  if (size_ == 0) return;
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (mapping == nullptr) {
    CloseHandle(file);
    file_ = nullptr;
    fail("cannot map " + path);
  }
  mapping_ = mapping;
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  if (view == nullptr) {
    CloseHandle(mapping);
    CloseHandle(file);
    mapping_ = nullptr;
    file_ = nullptr;
    fail("cannot map view of " + path);
  }
  data_ = static_cast<const uint8_t*>(view);
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fail("cannot open " + path);
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    fail("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);
  if (size_ > 0) {
    void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) {
      ::close(fd);
      fail("cannot map " + path);
    }
    data_ = static_cast<const uint8_t*>(p);
  }
  ::close(fd);
#endif
}

MappedFile::~MappedFile() {
#if defined(_WIN32)
  if (data_ != nullptr) UnmapViewOfFile(data_);
  if (mapping_ != nullptr) CloseHandle(static_cast<HANDLE>(mapping_));
  if (file_ != nullptr) CloseHandle(static_cast<HANDLE>(file_));
#else
  if (data_ != nullptr) ::munmap(const_cast<uint8_t*>(data_), size_);
#endif
}

size_t dtype_size(DType type) {
  switch (type) {
    case DType::Bool:
    case DType::U8:
    case DType::I8:
      return 1;
    case DType::U16:
    case DType::I16:
    case DType::F16:
    case DType::BF16:
      return 2;
    case DType::U32:
    case DType::I32:
    case DType::F32:
      return 4;
    case DType::U64:
    case DType::I64:
    case DType::F64:
      return 8;
  }
  return 0;
}

const char* dtype_name(DType type) {
  switch (type) {
    case DType::Bool:
      return "BOOL";
    case DType::U8:
      return "U8";
    case DType::I8:
      return "I8";
    case DType::U16:
      return "U16";
    case DType::I16:
      return "I16";
    case DType::F16:
      return "F16";
    case DType::BF16:
      return "BF16";
    case DType::U32:
      return "U32";
    case DType::I32:
      return "I32";
    case DType::F32:
      return "F32";
    case DType::U64:
      return "U64";
    case DType::I64:
      return "I64";
    case DType::F64:
      return "F64";
  }
  return "?";
}

bool parse_dtype(std::string_view name, DType& out) {
  static const DType all[] = {DType::Bool, DType::U8,   DType::I8,  DType::U16, DType::I16,
                              DType::F16,  DType::BF16, DType::U32, DType::I32, DType::F32,
                              DType::U64,  DType::I64,  DType::F64};
  for (DType t : all) {
    if (name == dtype_name(t)) {
      out = t;
      return true;
    }
  }
  return false;
}

int64_t TensorInfo::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

SafeTensors::SafeTensors(const std::string& path) : file_(std::make_unique<MappedFile>(path)) {
  const uint8_t* base = file_->data();
  size_t size = file_->size();
  if (size < 8) fail(path + ": file too small for safetensors");
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | base[i];
  if (header_len > size - 8 || header_len > (uint64_t{1} << 30)) fail(path + ": bad header length");
  std::string_view header(reinterpret_cast<const char*>(base + 8), static_cast<size_t>(header_len));
  Json doc = Json::parse(header);
  if (!doc.is_object()) fail(path + ": header is not an object");
  const uint8_t* data = base + 8 + header_len;
  uint64_t data_size = size - 8 - header_len;
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (size_t i = 0; i < doc.size(); ++i) {
    const std::string& key = doc.keys()[i];
    const Json& value = doc.items()[i];
    if (key == "__metadata__") {
      if (!value.is_object()) fail(path + ": metadata is not an object");
      for (size_t j = 0; j < value.size(); ++j) {
        if (!value.items()[j].is_string()) fail(path + ": metadata values must be strings");
        metadata_.emplace_back(value.keys()[j], value.items()[j].as_string());
      }
      continue;
    }
    TensorInfo t;
    t.name = key;
    if (!parse_dtype(value.get_string("dtype"), t.dtype)) fail(path + ": unknown dtype for " + key);
    const Json& shape = value["shape"];
    const Json& offsets = value["data_offsets"];
    if (!shape.is_array() || !offsets.is_array() || offsets.size() != 2) {
      fail(path + ": malformed entry for " + key);
    }
    int64_t numel = 1;
    for (const Json& d : shape.items()) {
      int64_t dim = d.as_int();
      if (dim < 0 || (dim > 0 && numel > std::numeric_limits<int64_t>::max() / dim)) {
        fail(path + ": bad shape for " + key);
      }
      t.shape.push_back(dim);
      numel *= dim;
    }
    int64_t begin = offsets.at(0).as_int();
    int64_t end = offsets.at(1).as_int();
    if (begin < 0 || end < begin || static_cast<uint64_t>(end) > data_size) {
      fail(path + ": data offsets out of range for " + key);
    }
    t.nbytes = static_cast<uint64_t>(end - begin);
    if (t.nbytes != static_cast<uint64_t>(numel) * dtype_size(t.dtype)) {
      fail(path + ": size mismatch for " + key);
    }
    t.data = data + begin;
    ranges.emplace_back(static_cast<uint64_t>(begin), static_cast<uint64_t>(end));
    index_[t.name] = tensors_.size();
    tensors_.push_back(std::move(t));
  }
  std::sort(ranges.begin(), ranges.end());
  for (size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].first < ranges[i - 1].second) fail(path + ": overlapping tensors");
  }
}

const TensorInfo* SafeTensors::find(std::string_view name) const {
  auto it = index_.find(std::string(name));
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

bool SafeTensors::has_metadata(std::string_view key) const {
  for (const auto& kv : metadata_) {
    if (kv.first == key) return true;
  }
  return false;
}

std::string SafeTensors::metadata(std::string_view key) const {
  for (const auto& kv : metadata_) {
    if (kv.first == key) return kv.second;
  }
  return {};
}

void write_safetensors(const std::string& path,
                       const std::vector<std::pair<std::string, std::string>>& metadata,
                       const std::vector<TensorSpec>& tensors, size_t alignment) {
  struct Entry {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    uint64_t begin;
    uint64_t end;
    const TensorSpec* spec;
  };
  std::vector<Entry> entries;
  uint64_t offset = 0;
  int pads = 0;
  for (const TensorSpec& t : tensors) {
    uint64_t numel = 1;
    for (int64_t d : t.shape) numel *= static_cast<uint64_t>(d);
    uint64_t nbytes = numel * dtype_size(t.dtype);
    if (offset % alignment != 0) {
      uint64_t pad = alignment - offset % alignment;
      entries.push_back({"__pad." + std::to_string(pads++),
                         DType::U8,
                         {static_cast<int64_t>(pad)},
                         offset,
                         offset + pad,
                         nullptr});
      offset += pad;
    }
    entries.push_back({t.name, t.dtype, t.shape, offset, offset + nbytes, &t});
    offset += nbytes;
  }
  Json header = Json::object();
  if (!metadata.empty()) {
    Json meta = Json::object();
    for (const auto& kv : metadata) meta.set(kv.first, Json(kv.second));
    header.set("__metadata__", std::move(meta));
  }
  for (const Entry& e : entries) {
    Json t = Json::object();
    t.set("dtype", Json(dtype_name(e.dtype)));
    Json shape = Json::array();
    for (int64_t d : e.shape) shape.push(Json(d));
    t.set("shape", std::move(shape));
    Json offsets = Json::array();
    offsets.push(Json(e.begin));
    offsets.push(Json(e.end));
    t.set("data_offsets", std::move(offsets));
    header.set(e.name, std::move(t));
  }
  std::string text = header.dump();
  while ((8 + text.size()) % alignment != 0) text += ' ';
  std::ofstream out(fs_path(path), std::ios::binary | std::ios::trunc);
  if (!out) fail("cannot create " + path);
  uint8_t len[8];
  uint64_t n = text.size();
  for (int i = 0; i < 8; ++i) len[i] = static_cast<uint8_t>(n >> (8 * i));
  out.write(reinterpret_cast<const char*>(len), 8);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  std::vector<uint8_t> buffer;
  for (const Entry& e : entries) {
    uint64_t nbytes = e.end - e.begin;
    buffer.assign(static_cast<size_t>(nbytes), 0);
    if (e.spec != nullptr && e.spec->fill) e.spec->fill(buffer.data(), nbytes);
    out.write(reinterpret_cast<const char*>(buffer.data()), static_cast<std::streamsize>(nbytes));
    if (!out) fail("write failed for " + path);
  }
  out.close();
  if (!out) fail("write failed for " + path);
}

}

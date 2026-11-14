cmake_minimum_required(VERSION 3.20)
project(llm_inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(INFER_WERROR "treat warnings as errors" OFF)
set(INFER_SANITIZE "" CACHE STRING "sanitizer to enable: address, undefined, thread")

if(MSVC)
  add_compile_options(/W4 /permissive- /utf-8 /EHsc /Zc:__cplusplus)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  if(INFER_WERROR)
    add_compile_options(/WX)
  endif()
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  if(INFER_WERROR)
    add_compile_options(-Werror)
  endif()
  if(MINGW)
    add_compile_options(-Wa,-muse-unaligned-vector-move)
    add_link_options(-static)
  endif()
endif()

if(INFER_SANITIZE)
  add_compile_options(-fsanitize=${INFER_SANITIZE} -fno-omit-frame-pointer)
  add_link_options(-fsanitize=${INFER_SANITIZE})
endif()

set(INFER_SOURCES
  src/common.cpp
  src/json.cpp
  src/safetensors.cpp
  src/kernels.cpp
  src/kernels_scalar.cpp
  src/quant.cpp
  src/thread_pool.cpp
  src/unicode.cpp
  src/unicode_tables.cpp
  src/tokenizer.cpp
  src/model.cpp
  src/runtime.cpp
  src/engine.cpp
  src/server.cpp
)

set(INFER_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64|X64)$")
  set(INFER_X86 ON)
endif()

if(INFER_X86)
  list(APPEND INFER_SOURCES src/kernels_avx2.cpp src/kernels_vnni.cpp src/kernels_avx512.cpp)
  if(MSVC)
    set_source_files_properties(src/kernels_avx2.cpp src/kernels_vnni.cpp
      PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/kernels_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(src/kernels_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
    set_source_files_properties(src/kernels_vnni.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mavxvnni")
    set_source_files_properties(src/kernels_avx512.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni")
  endif()
endif()

find_package(Threads REQUIRED)

add_library(infer_core STATIC ${INFER_SOURCES})
target_include_directories(infer_core PUBLIC src)
target_link_libraries(infer_core PUBLIC Threads::Threads)
if(INFER_X86)
  target_compile_definitions(infer_core PUBLIC INFER_X86=1)
endif()
if(WIN32)
  target_link_libraries(infer_core PUBLIC ws2_32)
  target_compile_definitions(infer_core PUBLIC _WIN32_WINNT=0x0A00)
endif()

add_executable(infer src/main.cpp)
target_link_libraries(infer PRIVATE infer_core)
if(WIN32)
  target_link_libraries(infer PRIVATE shell32)
endif()

enable_testing()
foreach(name kernels tokenizer model server)
  add_executable(${name}_test tests/${name}_test.cpp)
  target_link_libraries(${name}_test PRIVATE infer_core)
  add_test(NAME ${name} COMMAND ${name}_test)
endforeach()

#!/bin/bash
# Builds doom-wasm inside the emsdk container (emcc not installed locally).
set -e
cd "$(dirname "$0")/doom-wasm"
docker run --rm -v doom-emcache:/emsdk/upstream/emscripten/cache -v "$PWD":/src -w /src emscripten/emsdk:latest bash -c '
  set -e
  apt-get update -qq >/dev/null && apt-get install -y -qq automake autoconf pkg-config >/dev/null
  emmake make clean >/dev/null 2>&1 || true
  emconfigure autoreconf -fiv
  ac_cv_exeext=".html" emconfigure ./configure --host=none-none-none
  emmake make -j8
'

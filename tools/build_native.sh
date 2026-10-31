#!/bin/sh
# Build the native core and drop it next to the package for in-tree runs.
set -e
cd "$(dirname "$0")/.."
cargo build --release -p moses-py
cp target/release/lib_core.so src/vagrant_moses/_core.abi3.so

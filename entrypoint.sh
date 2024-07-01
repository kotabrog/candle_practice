#!/bin/bash

TARGET_DIR=$1

# ディレクトリが存在するか確認
if [ -d "/workspace/sample/$TARGET_DIR" ]; then
  cd "/workspace/sample/$TARGET_DIR"
  echo "Building in /workspace/sample/$TARGET_DIR"
  cargo build --release
  echo "Running in /workspace/sample/$TARGET_DIR"
  cargo run --release
else
  echo "Directory /workspace/sample/$TARGET_DIR does not exist."
  exit 1
fi

if [ "$2" == "interactive" ]; then
  exec /bin/bash
fi

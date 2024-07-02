#!/bin/bash

TARGET_EXAMPLE=$1

# ディレクトリが存在するか確認
if [ -d "/workspace/tutorial" ]; then
  echo "cd /workspace/tutorial"
  cd "/workspace/tutorial"
  echo "Building $TARGET_EXAMPLE"
  cargo build --release --example $TARGET_EXAMPLE
  echo "Running $TARGET_EXAMPLE"
  cargo run --release --example $TARGET_EXAMPLE
else
  echo "Directory /workspace/tutorial does not exist."
  exit 1
fi

if [ "$2" == "interactive" ]; then
  exec /bin/bash
fi

#!/bin/bash

CRATE=$1
TARGET_EXAMPLE=$2
INTERACTIVE=$3

# ディレクトリが存在するか確認
if [ -d "/workspace/$CRATE" ]; then
  echo "cd /workspace/$CRATE"
  cd "/workspace/$CRATE"
  if [ "$TARGET_EXAMPLE" == "test" ]; then
    echo "Testing $CRATE"
    cargo test --release
  else
    echo "Building $TARGET_EXAMPLE"
    cargo build --release --example $TARGET_EXAMPLE
    echo "Running $TARGET_EXAMPLE"
    cargo run --release --example $TARGET_EXAMPLE
  fi
else
  echo "Directory /workspace/$CRATE does not exist."
  exit 1
fi

if [ "$INTERACTIVE" == "interactive" ]; then
  exec /bin/bash
fi

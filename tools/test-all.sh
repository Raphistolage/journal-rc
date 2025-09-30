#!/bin/bash
# Test all examples in the repository

set -e

echo "Testing all examples..."

cd "$(dirname "$0")/.."

for example in examples/*/; do
    if [ -f "$example/Cargo.toml" ]; then
        echo "Testing $example..."
        cd "$example"
        cargo test --quiet
        cd - > /dev/null
    fi
done

echo "All tests passed!"
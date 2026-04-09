#!/bin/bash

for i in $(seq 1 20); do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"mlx-community/Qwen3-0.6B-4bit\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Tell me fact number $i about distributed systems.\"}],
      \"max_tokens\": 100
    }" > /dev/null &
done
wait
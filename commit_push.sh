#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  printf 'Usage: %s "commit message"\n' "$(basename "$0")" >&2
  exit 64
fi

message="$*"

git add .
git commit -m "$message"
git push

#!/usr/bin/env bash
# Build the gd example (via top-level `make`) and run it with several
# flag combinations, asserting each run drives the iterate norm below
# a convergence threshold. Exercises the positional CLI surface.
set -euo pipefail

GD="${GD:-out/gd}"
THRESHOLD="${THRESHOLD:-1e-3}"

run_case() {
  local label="$1"; shift
  local log
  log="$("$GD" "$@")"
  local err
  err="$(printf '%s\n' "$log" | awk '/^Iter:/ {e=$NF} END {print e}')"
  printf '[gd] %-30s args=%-40s err=%s\n' "$label" "$*" "$err"
  awk -v e="$err" -v t="$THRESHOLD" 'BEGIN {
    if (e+0 >= t+0) { printf "gd did not converge: %s >= %s\n", e, t; exit 1 }
  }'
}

#         label                       mem type1 dim  step  seed iters [reg]
run_case "type1 mem=5"                  5  1   100  0.01   0   3000
run_case "type2 mem=5"                  5  0   100  0.01   0   3000
run_case "type1 mem=10 reg=1e-8"       10  1   100  0.01   0   3000  1e-8

#!/usr/bin/env bash
# Build the gd example (via top-level `make`) and run it with several
# flag combinations, asserting each run drives the iterate norm below
# a convergence threshold. Exercises the positional CLI surface.
#
# Optional env vars:
#   GD        path to the gd binary (default: out/gd)
#   RUNNER    command prefix (e.g. "valgrind --error-exitcode=1 -q")
#   THRESHOLD convergence cutoff for the final iterate norm (default: 1e-3)
set -euo pipefail

GD="${GD:-out/gd}"
THRESHOLD="${THRESHOLD:-1e-3}"
# shellcheck disable=SC2206  # intentional word-splitting for the runner prefix
RUNNER_CMD=( ${RUNNER:-} )

run_case() {
  local label="$1"; shift
  local log
  log="$("${RUNNER_CMD[@]+"${RUNNER_CMD[@]}"}" "$GD" "$@")"
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

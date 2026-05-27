#!/usr/bin/env bash
#
# Restartable orchestrator for the v3 bundled-personas anchoring run.
#
# Picks up where the prior run left off (after the 26-concept vector
# regen completed but before neutrals + persona corpora + anchored
# manifold fit landed):
#
#   stage 1: neutrals regen (writes saklas/data/neutral_statements.json,
#            the corpus the anchor node sources at stage 2)
#   stage 2: persona manifold corpora regen (101 nodes — 100 personas
#            + 1 anchor sourced from the fresh neutrals; hyperparams
#            carry `anchor_origin: true`)
#   stage 3: vector extraction loop (re-extracts all 26 vectors against
#            fresh neutrals; mu_neutral changes -> DLS layer set may
#            shift, baked tensors refresh)
#   stage 4: personas manifold discover-fit (picks up anchor_origin from
#            the manifest, translates origin onto the anchor node)
#
# Logs to /tmp/saklas-anchor/run.log with a /tmp/saklas-anchor/done
# sentinel carrying per-stage exit codes.

set -u
set -o pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="/tmp/saklas-anchor"
LOG_FILE="$LOG_DIR/run.log"
DONE_FILE="$LOG_DIR/done"
MODEL="${SAKLAS_REGEN_MODEL:-google/gemma-4-31b-it}"

mkdir -p "$LOG_DIR"
rm -f "$DONE_FILE" "$LOG_FILE"

exec > "$LOG_FILE" 2>&1

echo "=== started $(date) ==="
echo "repo: $REPO"
echo "model: $MODEL"

cd "$REPO"

echo ""
echo "=== STAGE 1/4: neutrals regen ==="
date
python3 scripts/regenerate_bundled_statements.py --only-neutrals
s1=$?
echo "--- stage 1 exit=$s1 at $(date) ---"

if [ "$s1" -ne 0 ]; then
    echo "stage 1 failed — aborting before stage 2 (it depends on the fresh neutrals file)"
    echo "stage_exits: s1=$s1 s2=skipped s3=skipped s4=skipped" > "$DONE_FILE"
    exit "$s1"
fi

echo ""
echo "=== STAGE 2/4: persona manifold corpora regen ==="
date
python3 scripts/regenerate_bundled_manifold.py --force
s2=$?
echo "--- stage 2 exit=$s2 at $(date) ---"

echo ""
echo "=== STAGE 3/4: vector extraction loop ==="
date
s3=0
# Explicit `default/<c>` namespacing: bare names hit
# ``AmbiguousSelectorError`` when the user keeps a sibling namespace
# (`default-archive/`, `local/`, etc.) populated with the same concept
# slugs.  The bundled vectors always live under `default/`, so qualify
# at the source rather than depend on disambiguation.
for c in $(ls saklas/data/vectors); do
    echo ""
    echo "--- extracting: default/$c ---"
    saklas vector extract "default/$c" -m "$MODEL" -f
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "vector extract failed for default/$c (rc=$rc) — continuing"
        s3="$rc"
    fi
done
echo "--- stage 3 finished at $(date) (worst rc=$s3) ---"

echo ""
echo "=== STAGE 4/4: personas manifold discover-fit (anchored) ==="
date
saklas vector manifold discover default/personas -m "$MODEL"
s4=$?
echo "--- stage 4 exit=$s4 at $(date) ---"

echo ""
echo "=== finished $(date) ==="
echo "stage_exits: s1=$s1 s2=$s2 s3=$s3 s4=$s4" > "$DONE_FILE"

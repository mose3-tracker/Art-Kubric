#!/bin/bash
# Batch-compile Articraft-10K records into URDF + OBJ assets for the Art-Kubric
# pipeline. Run manually once (executes the articraft SDK compiler; a few
# seconds per record on a login/CPU node).
#
# Usage: bash tools/compile_articraft_batch.sh [NUM_ASSETS] [MAX_PER_CATEGORY]
#   NUM_ASSETS       total records to compile (default 300)
#   MAX_PER_CATEGORY per-category cap for diversity (default 3)
#
# Env overrides: WORK (scratch dir), ASSETS_OUT (destination), PYBIN.
#
# Requirements:
#   - articraft needs python >=3.11,<3.13
#   - cadquery from conda-forge; set ARTICRAFT_PY=/path/to/conda-env/bin/python
#     (pip's cadquery-ocp wheels need glibc >= 2.31)
set -euo pipefail

NUM_ASSETS="${1:-300}"
MAX_PER_CATEGORY="${2:-3}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$REPO_DIR/work_articraft}"
ASSETS_OUT="${ASSETS_OUT:-${SAPIEN_ASSETS_ROOT:?set SAPIEN_ASSETS_ROOT or ASSETS_OUT}/articraft}"
mkdir -p "$WORK" "$ASSETS_OUT"

# 1. Clone SDK + data repos (data repo is code/JSON only, no large binaries)
[ -d "$WORK/articraft" ] || git clone --depth 1 https://github.com/mattzh72/articraft.git "$WORK/articraft"
[ -d "$WORK/articraft-data" ] || git clone --depth 1 https://github.com/mattzh72/articraft-data.git "$WORK/articraft-data"

# 2. Environment (python >=3.11,<3.13; no cadquery -- see header)
PYBIN="${PYBIN:-}"
if [ -z "$PYBIN" ]; then
    for cand in python3.12 python3.11 /usr/bin/python3.12 /usr/bin/python3.11; do
        if command -v "$cand" >/dev/null 2>&1; then PYBIN="$cand"; break; fi
    done
fi
[ -n "$PYBIN" ] || { echo "Error: need python 3.11/3.12"; exit 1; }
VENV_PY="${ARTICRAFT_PY:-$WORK/articraft/.venv/bin/python}"
if [ -n "${ARTICRAFT_PY:-}" ]; then
    # pip wheels (numpy etc.) in the conda env need the env's own libstdc++,
    # not RHEL8's /lib64 one (GLIBCXX_3.4.29).
    export LD_LIBRARY_PATH="$(dirname "$(dirname "$ARTICRAFT_PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [ -z "${ARTICRAFT_PY:-}" ] && [ -x "$VENV_PY" ] && ! "$VENV_PY" -c 'import manifold3d' 2>/dev/null; then
    rm -rf "$WORK/articraft/.venv"
fi
if [ -z "${ARTICRAFT_PY:-}" ] && [ ! -x "$VENV_PY" ]; then
    "$PYBIN" -m venv "$WORK/articraft/.venv"
    "$WORK/articraft/.venv/bin/pip" install -q \
        aiofiles fastapi google-genai numpy openai python-dotenv rich uvicorn \
        pydantic manifold3d trimesh python-fcl websockets networkx rtree \
        zstandard bm25s scipy anthropic socksio
    "$WORK/articraft/.venv/bin/pip" install -q --no-deps -e "$WORK/articraft"
fi

# 3. Pick top-rated records: rating=5, success, non-cadquery, capped per category
PICKS=$("$VENV_PY" - "$WORK/articraft-data" "$NUM_ASSETS" "$MAX_PER_CATEGORY" <<'EOF'
import json, sys
from collections import Counter
from pathlib import Path
data_root, n, per_cat = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cat_counts, picks = Counter(), []
rows = [json.loads(l) for l in (data_root / "records_manifest.jsonl").open()]
rows.sort(key=lambda r: (-(r.get("rating") or 0), r["record_id"]))
for r in rows:
    if r.get("run_status") != "success" or (r.get("rating") or 0) < 5:
        continue
    if cat_counts[r["category_slug"]] >= per_cat:
        continue
    model = data_root / "records" / r["record_id"] / "revisions" / r["active_revision_id"] / "model.py"
    if not model.exists():
        continue
    cat_counts[r["category_slug"]] += 1
    picks.append(f'{r["record_id"]}\t{r["category_slug"]}')
    if len(picks) >= n:
        break
print("\n".join(picks))
EOF
)

# 4. Compile each record (model.py -> model.urdf + assets/meshes/*.obj)
cd "$WORK/articraft"
OK=0; FAIL=0; SKIP=0
FAILED_IDS="$WORK/failed_records.txt"; : > "$FAILED_IDS"
while IFS=$'\t' read -r rid cat; do
    dest="$ASSETS_OUT/${cat}__${rid:0:40}"
    if [ -f "$dest/model.urdf" ]; then
        SKIP=$((SKIP + 1)); continue
    fi
    echo "=== compiling $cat ($rid)"
    if "$VENV_PY" -m cli.main compile "$rid" --data-dir "$WORK/articraft-data" >/dev/null 2>&1; then
        mat="$WORK/articraft-data/cache/record_materialization/$rid"
        # A "successful" compile can still produce no usable output (e.g. no
        # assets dir). Guard every copy: one bad record must not abort the
        # batch (set -e), it just goes on the failed list.
        if [ -f "$mat/model.urdf" ] && [ -d "$mat/assets" ] \
           && rm -rf "$dest" && mkdir -p "$dest" \
           && cp "$mat/model.urdf" "$dest/" \
           && cp -r "$mat/assets" "$dest/"; then
            OK=$((OK + 1))
        else
            echo "!!! compile produced no usable output: $rid"
            echo "$rid" >> "$FAILED_IDS"
            FAIL=$((FAIL + 1))
            rm -rf "$dest"
        fi
    else
        echo "!!! compile failed: $rid"
        echo "$rid" >> "$FAILED_IDS"
        FAIL=$((FAIL + 1))
    fi
done <<< "$PICKS"

# 5. Sanitize inertials (degenerate tensors are rejected by SAPIEN)
"$VENV_PY" "$REPO_DIR/tools/sanitize_urdf_inertials.py" "$ASSETS_OUT"

echo "Done: compiled $OK, failed $FAIL (see $FAILED_IDS), skipped-existing $SKIP"
echo "Assets in $ASSETS_OUT"

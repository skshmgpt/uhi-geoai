#!/usr/bin/env bash
# E5 hotspot stability: load every year x threshold candidate set, build hotspots, compare top-20 sets.
# Usage: scripts/e5_stability.sh   (writes $RESULTS/e5_stability.json)
set -euo pipefail
cd "$(dirname "$0")/.."
RESULTS=${RESULTS:-pipeline/results}
YEARS=${YEARS:-"2023 2024 2025"}
THRESHOLDS=${THRESHOLDS:-"1.5 2.0 2.5"}
MIN_A=${MIN_AREA:-5000}
TOP=${TOP_N:-20}
psql() { docker compose exec -T db psql -v ON_ERROR_STOP=1 -q -U uhi -d uhi "$@"; }

psql < db/03_stability.sql
for y in $YEARS; do
  for t in $THRESHOLDS; do
    psql < "$RESULTS/hotspot_candidates_${y}_${t}.sql"
    echo "$y I>=$t: $(psql -At -c "SELECT build_hotspots($y, $t, $MIN_A);") hotspots"
  done
done

# Pairs: each year pair at 2.0 C, and each threshold against 2.0 C within a year.
pairs=()
set -- $YEARS
for a in "$@"; do for b in "$@"; do if [[ $a < $b ]]; then pairs+=("$a 2.0 $b 2.0"); fi; done; done
for y in $YEARS; do for t in $THRESHOLDS; do if [[ $t != 2.0 ]]; then pairs+=("$y 2.0 $y $t"); fi; done; done

{
  echo '{"top_n": '"$TOP"', "min_area_m2": '"$MIN_A"', "pairs": ['
  sep=""
  for p in "${pairs[@]}"; do
    read -r y1 i1 y2 i2 <<<"$p"
    row=$(psql -At -F ' ' -c "SELECT round(hotspot_jaccard($y1, $i1, $MIN_A, $y2, $i2, $MIN_A, $TOP)::numeric, 4),
                                     round(s.rho::numeric, 4), s.n
                              FROM hotspot_rank_spearman($y1, $i1, $MIN_A, $y2, $i2, $MIN_A, $TOP) s;")
    read -r jac rho n <<<"$row"
    printf '%s  {"a": [%s, %s], "b": [%s, %s], "jaccard": %s, "spearman": %s, "matched": %s}\n' \
      "$sep" "$y1" "$i1" "$y2" "$i2" "${jac:-null}" "${rho:-null}" "${n:-0}"
    sep=","
  done
  echo ']}'
} > "$RESULTS/e5_stability.json"
cat "$RESULTS/e5_stability.json"

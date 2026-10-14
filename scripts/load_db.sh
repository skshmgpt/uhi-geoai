#!/usr/bin/env bash
# Load one year's pipeline outputs into PostGIS.  Usage: scripts/load_db.sh 2024
set -euo pipefail
YEAR=${1:?usage: load_db.sh YEAR}
MIN_I=${MIN_INTENSITY:-2.0}
MIN_A=${MIN_AREA:-5000}
cd "$(dirname "$0")/.."
RESULTS=${RESULTS:-pipeline/results}
psql() { docker compose exec -T db psql -v ON_ERROR_STOP=1 -q -U uhi -d uhi "$@"; }

for r in ndvi30_study builtfrac30; do
  table=${r%_study}
  docker compose exec -T db raster2pgsql -d -s 32643 -t 100x100 -I -C "/data/${r}_${YEAR}.tif" "$table" | psql
  echo "loaded raster $table"
done

psql < "$RESULTS/hotspot_candidates_${YEAR}_${MIN_I}.sql"
psql -c "SELECT build_hotspots($YEAR, $MIN_I, $MIN_A) AS hotspots;"

params=$(cat "$RESULTS/cooling_params_${YEAR}.json")
psql <<SQL
INSERT INTO cooling_params (year, params) VALUES ($YEAR, \$json\$${params}\$json\$::jsonb)
ON CONFLICT (year) DO UPDATE SET params = EXCLUDED.params, loaded_at = now();
SQL
echo "loaded cooling params for $YEAR"

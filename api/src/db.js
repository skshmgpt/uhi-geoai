// PostGIS queries. All geometry work happens in EPSG:32643 (metres); I/O is WGS84 GeoJSON.
import pg from 'pg';

const UTM = 32643;

export function createDb(connectionString) {
  const pool = new pg.Pool({ connectionString, max: 10 });
  const one = async (sql, args) => (await pool.query(sql, args)).rows[0];

  return {
    close: () => pool.end(),

    async params(year) {
      const row = await one('SELECT params FROM cooling_params WHERE year = $1', [year]);
      return row?.params ?? null;
    },

    async hotspots(year, { minIntensity = 2.0, minArea = 5000, limit = 50 } = {}) {
      const { rows } = await pool.query(
        `SELECT rank, mean_intensity, area_m2, heat_load,
                ST_AsGeoJSON(ST_Transform(geom, 4326), 6)::json AS geometry
         FROM hotspots
         WHERE year = $1 AND min_intensity = $2 AND min_area_m2 = $3 AND rank <= $4
         ORDER BY rank`,
        [year, minIntensity, minArea, limit],
      );
      return {
        type: 'FeatureCollection',
        features: rows.map(({ geometry, ...properties }) => ({ type: 'Feature', geometry, properties })),
      };
    },

    // Traits of a drawn polygon: area, Fragstats SHAPE, current NDVI, built fraction in a 300 m ring.
    // Small polygons can miss every raster cell centre, so fall back to the value at the centroid.
    async traits(geojson) {
      return one(
        `WITH g AS (SELECT ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326), ${UTM}) AS geom),
              ring AS (SELECT ST_Difference(ST_Buffer(geom, 300), geom) AS geom FROM g)
         SELECT ST_Area(g.geom) AS area_m2,
                0.25 * ST_Perimeter(g.geom) / sqrt(ST_Area(g.geom)) AS shape,
                ST_IsValid(g.geom) AS valid,
                COALESCE(
                  (SELECT (ST_SummaryStatsAgg(ST_Clip(r.rast, g.geom), 1, true, 1)).mean
                     FROM ndvi30 r WHERE ST_Intersects(r.rast, g.geom)),
                  (SELECT ST_Value(r.rast, ST_Centroid(g.geom)) FROM ndvi30 r
                     WHERE ST_Intersects(r.rast, ST_Centroid(g.geom)) LIMIT 1)
                ) AS ndvi_now,
                (SELECT (ST_SummaryStatsAgg(ST_Clip(r.rast, ring.geom), 1, true, 1)).mean
                   FROM builtfrac30 r WHERE ST_Intersects(r.rast, ring.geom)) AS built
         FROM g, ring`,
        [JSON.stringify(geojson)],
      );
    },

    // Concentric rings (outer distance d) around the polygon, as WGS84 GeoJSON.
    async rings(geojson, stepM, maxM) {
      const { rows } = await pool.query(
        `WITH g AS (SELECT ST_Transform(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326), ${UTM}) AS geom)
         SELECT d::float AS outer_m,
                ST_AsGeoJSON(ST_Transform(ST_Difference(ST_Buffer(g.geom, d), ST_Buffer(g.geom, d - $2)), 4326), 6)::json AS geometry
         FROM g, generate_series($2::numeric, $3::numeric, $2::numeric) AS d`,
        [JSON.stringify(geojson), stepM, maxM],
      );
      return rows;
    },
  };
}

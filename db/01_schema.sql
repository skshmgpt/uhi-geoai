-- Schema for the online service. Geometries are UTM 43N (EPSG:32643, metres);
-- the API converts to and from WGS84 at the edges.
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;

-- Connected regions above the SUHI threshold, written by `python -m uhi.process hotspots`.
CREATE TABLE IF NOT EXISTS hotspot_candidates (
  id serial PRIMARY KEY,
  year int NOT NULL,
  min_intensity real NOT NULL,
  mean_intensity real NOT NULL,
  area_m2 real NOT NULL,
  geom geometry(Polygon, 32643) NOT NULL
);
CREATE INDEX IF NOT EXISTS hotspot_candidates_geom ON hotspot_candidates USING gist (geom);

-- Merged, filtered and ranked hotspots (build_hotspots below).
CREATE TABLE IF NOT EXISTS hotspots (
  id serial PRIMARY KEY,
  year int NOT NULL,
  min_intensity real NOT NULL,
  min_area_m2 real NOT NULL,
  rank int NOT NULL,
  mean_intensity real NOT NULL,
  area_m2 real NOT NULL,
  heat_load real NOT NULL,
  geom geometry(MultiPolygon, 32643) NOT NULL
);
CREATE INDEX IF NOT EXISTS hotspots_geom ON hotspots USING gist (geom);
CREATE INDEX IF NOT EXISTS hotspots_lookup ON hotspots (year, min_intensity, min_area_m2, rank);

-- Fitted cooling model per year (`python -m uhi.process calibrate`).
CREATE TABLE IF NOT EXISTS cooling_params (
  year int PRIMARY KEY,
  params jsonb NOT NULL,
  loaded_at timestamptz NOT NULL DEFAULT now()
);

-- Rasters ndvi30 and builtfrac30 (30 m, study box) are created by raster2pgsql in scripts/load_db.sh.

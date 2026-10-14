-- Merge candidate fragments within eps metres, drop clusters under p_min_area, rank by heat load.
CREATE OR REPLACE FUNCTION build_hotspots(p_year int, p_min_intensity real, p_min_area real, p_eps real DEFAULT 60)
RETURNS int LANGUAGE sql AS $$
  DELETE FROM hotspots WHERE year = p_year AND min_intensity = p_min_intensity AND min_area_m2 = p_min_area;
  WITH c AS (
    SELECT ST_ClusterDBSCAN(geom, eps := p_eps, minpoints := 1) OVER () AS cid, geom, mean_intensity, area_m2
    FROM hotspot_candidates
    WHERE year = p_year AND min_intensity = p_min_intensity
  ), agg AS (
    SELECT ST_Multi(ST_Union(geom)) AS geom,
           sum(mean_intensity * area_m2) / sum(area_m2) AS mi,
           sum(area_m2) AS a
    FROM c GROUP BY cid HAVING sum(area_m2) >= p_min_area
  )
  INSERT INTO hotspots (year, min_intensity, min_area_m2, rank, mean_intensity, area_m2, heat_load, geom)
  SELECT p_year, p_min_intensity, p_min_area, rank() OVER (ORDER BY mi * a DESC), mi, a, mi * a, geom FROM agg;
  SELECT count(*)::int FROM hotspots WHERE year = p_year AND min_intensity = p_min_intensity AND min_area_m2 = p_min_area;
$$;

-- E5: area-based Jaccard overlap of the top-n hotspots between two hotspot sets.
CREATE OR REPLACE FUNCTION hotspot_jaccard(
  y1 int, i1 real, a1 real, y2 int, i2 real, a2 real, top_n int DEFAULT 20
) RETURNS double precision LANGUAGE sql AS $$
  WITH s1 AS (SELECT ST_Union(geom) g FROM hotspots WHERE year = y1 AND min_intensity = i1 AND min_area_m2 = a1 AND rank <= top_n),
       s2 AS (SELECT ST_Union(geom) g FROM hotspots WHERE year = y2 AND min_intensity = i2 AND min_area_m2 = a2 AND rank <= top_n)
  SELECT ST_Area(ST_Intersection(s1.g, s2.g)) / NULLIF(ST_Area(ST_Union(s1.g, s2.g)), 0) FROM s1, s2;
$$;

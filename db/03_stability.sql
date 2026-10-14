-- E5: rank agreement of the top-n hotspots of set 1 with their best-overlapping match in set 2.
-- Spearman's rho is the Pearson correlation of the two rank lists; unmatched set-1 hotspots are dropped,
-- and the returned n says how many were matched.
CREATE OR REPLACE FUNCTION hotspot_rank_spearman(
  y1 int, i1 real, a1 real, y2 int, i2 real, a2 real, top_n int DEFAULT 20
) RETURNS TABLE (rho double precision, n int) LANGUAGE sql AS $$
  WITH s1 AS (SELECT rank, geom FROM hotspots WHERE year = y1 AND min_intensity = i1 AND min_area_m2 = a1 AND rank <= top_n),
       s2 AS (SELECT rank, geom FROM hotspots WHERE year = y2 AND min_intensity = i2 AND min_area_m2 = a2),
       m AS (
         SELECT DISTINCT ON (s1.rank) s1.rank AS r1, s2.rank AS r2
         FROM s1 JOIN s2 ON ST_Intersects(s1.geom, s2.geom)
         ORDER BY s1.rank, ST_Area(ST_Intersection(s1.geom, s2.geom)) DESC
       ),
       rr AS (SELECT rank() OVER (ORDER BY r1) AS q1, rank() OVER (ORDER BY r2) AS q2 FROM m)
  SELECT corr(q1, q2), count(*)::int FROM rr;
$$;

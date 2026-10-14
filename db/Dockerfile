# PostGIS plus the raster2pgsql loader (the stock image ships only the extension).
# The pgdg repo for this Debian release is retired, so install from Debian's own repo.
FROM postgis/postgis:16-3.4
RUN rm -f /etc/apt/sources.list.d/pgdg.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends postgis \
 && rm -rf /var/lib/apt/lists/*

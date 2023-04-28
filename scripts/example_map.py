import folium
from folium import Map, plugins, Marker
from folium.plugins import MousePosition
import pandas as pd
import geohash_hilbert as gh
from omegaconf import DictConfig
from geohash_stupino import add_geohash, add_geohash_grid
from draw import add2map

city_center = {"lat": 54.885288, "lon": 38.087027}

basic_map = Map(
    location=[city_center.lat, city_center.lon],
    zoom_start=12,
    tiles='OpenStreetMap',
    control_scale=True,
    prefer_canvas=True,
    )




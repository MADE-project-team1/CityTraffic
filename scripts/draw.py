import folium
from folium import Map, plugins, Marker
from folium.plugins import MousePosition
import pandas as pd
import numpy as np
import time
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
import datetime
import os
import geohash_hilbert as gh

import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import logging

def add2map(map, latlon, cluster_id_id, cluster_number=-1, is_cluster_point=False, color='blue', group=None, points_of_cluster=None):
    if is_cluster_point:
        icon_obj = plugins.BeautifyIcon(
            icon='arrow-down', icon_shape='marker',
            border_color=color, text_color=color,
            number=cluster_id_id
        )
        popup_message = f"cluster: {cluster_number}\ncluster_size: {points_of_cluster}"
        marker = Marker(location=[latlon[0], latlon[1]], tooltip='cluster:' + str(cluster_number),
                popup=popup_message, icon=icon_obj)
    else:
        marker = Marker(location=[latlon[0], latlon[1]], tooltip='cluster:' + str(cluster_number),
                popup='cluster: ' + str(cluster_number))

    if group:
        marker.add_to(group)
    else:
        marker.add_to(map)

def add_geohash(map, hash, bits_per_char):
    rect = gh.rectangle(hash, bits_per_char=bits_per_char)
    bounds = rect['geometry']['coordinates'][0]
    folium.Polygon(
        bounds,
        color='blue',
        fill_color='white',
        weight=2,
        popup=hash,
        fill_opacity=0.5
    ).add_to(map)


def draw_map(save_data_folder: str, launch_time:str, cfg: DictConfig, log=None):
    ids_clusters_df = pd.read_csv(save_data_folder + f"/clusters_{launch_time}.csv");
    clusterised_locs = pd.read_csv(save_data_folder + f"/data_with_clusters_{launch_time}.csv");
    ids_clusters_df['geohash'] = ids_clusters_df['geohash'].astype(str)

    city_center = cfg.city_center

    if cfg.draw_map:
        basic_map = Map(
            location=[city_center.lat, city_center.lon],
            zoom_start=12,
            tiles='OpenStreetMap',
            control_scale=True,
            prefer_canvas=True,
            )
        
        Measure_Control = plugins.MeasureControl(
            primary_length_unit='meters', 
            primary_area_unit='sqmeters', 
        )
    else:
        basic_map = None

    ID_LIST = pd.unique(ids_clusters_df['id'])
    for cur_id in ID_LIST:
        rs = ids_clusters_df[ids_clusters_df['id'] == cur_id]
        locs_cur_id =  clusterised_locs[clusterised_locs['id'] == cur_id]
        if cfg.draw_map:
            if rs is not None:
                map_clusters = folium.FeatureGroup(name=f"clusters of id  {cur_id} ({len(rs)})", show=False)
                basic_map.add_child(map_clusters)
                for index, row in rs.iterrows():
                    lat, lon = float(row['lat']), float(row['lon'])
                    cluster_number = int(row['cluster'])
                    # log.info(rs)
                    # log.info(f'{cluster_number=}')
                    points_of_cluster = row['cluster_size']
                    add2map(basic_map, [lat, lon],  int(cur_id), cluster_number=cluster_number, group=map_clusters,
                            is_cluster_point=True, points_of_cluster=points_of_cluster)
                    add_geohash(basic_map, row['geohash'], bits_per_char=cfg.gh_bits_per_char)

            all_locations = folium.FeatureGroup(name=f"locs of id {cur_id} ({len(locs_cur_id)})" , show=False)
            basic_map.add_child(all_locations)
            for index, row in locs_cur_id.iterrows():
                lat, lon = float(row['lat']), float(row['lon'])
                cluster_number = int(row['cluster'])
                add2map(basic_map, [lat, lon], int(cur_id), cluster_number=cluster_number, group=all_locations)

    if cfg.draw_map:
        folium.LayerControl().add_to(basic_map);
        MousePosition().add_to(basic_map);
        basic_map.save(save_data_folder + f"/map_{launch_time}.html")

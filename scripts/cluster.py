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

import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import logging

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def add2map(map, latlon, cluster_id_id, cluster_number=-1, is_cluster_point=False, color='red', group=None, points_of_cluster=None):
    if is_cluster_point:
        icon_obj = plugins.BeautifyIcon(
            icon='arrow-down', icon_shape='marker',
            border_color=color, text_color=color,
            number=cluster_id_id
        )
        popup_message = f"cluster: {cluster_number}\npoints: {points_of_cluster}"
        marker = Marker(location=[latlon[0], latlon[1]], tooltip='cluster:' + str(cluster_number),
                popup=popup_message, icon=icon_obj)
    else:
        marker = Marker(location=[latlon[0], latlon[1]], tooltip='cluster:' + str(cluster_number),
                popup='cluster: ' + str(cluster_number))

    if group:
        marker.add_to(group)
    else:
        marker.add_to(map)

METERS_PER_RADIAN = 6371008.8
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def make_clusters(cfg: DictConfig):
    launch_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_data_folder = f"./data/{launch_time}"
    os.mkdir(save_data_folder)
    rng = np.random.default_rng(seed=cfg.seed)

    locs = pd.read_csv(cfg.locs_path)
    interests = pd.read_csv(cfg.interests_path)


    if isinstance(cfg.id_points_range, ListConfig):
        bounds = OmegaConf.to_container(cfg.id_points_range)
    else:
        if cfg.id_points_range >= 0:
            bounds = [0, cfg.id_points_range]
        else:
            bounds = [0, np.inf]

    unique_id = pd.unique(locs['id'])
    ids = unique_id[locs['id'].value_counts().between(*bounds)]

    if cfg.dist_policy == 'const':
        MAX_DIST_M = cfg.max_dist

    if cfg.sample_policy == 'const':
        MIN_SAMPLES = cfg.min_samples

    if cfg.n_ids == -1:
        id_LIST = ids
    elif isinstance(cfg.n_ids, ListConfig):
        id_LIST = OmegaConf.to_container(cfg.n_ids)
    else:
        id_LIST = rng.choice(ids, size=cfg.n_ids)

    epsilon = MAX_DIST_M / METERS_PER_RADIAN
    x = 'lon'
    y = 'lat'

    city_center = locs[[y, x]].mean()

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

    start_global_time = time.time()

    ids_clusters_df = pd.DataFrame({'id' : [], 'lat' : [], 'lon' : [], 'cluster' : []})
    clusterised_locs = pd.DataFrame(columns=locs.columns)
    clusterised_locs['is_weekday'] = clusterised_locs['is_weekday'].astype(bool)
    clusterised_locs['cluster'] = pd.Series(dtype='int')

    for cur_id in id_LIST:
        locs_cur_id = locs[locs['id'] == cur_id].copy()
        start_time = time.time()
        coords = locs_cur_id[[y, x]].values 
        db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, **cfg.model_params).fit(np.radians(coords))
        cluster_labels = db.labels_
        locs_cur_id['cluster'] = cluster_labels
        num_clusters = len(set(cluster_labels)) - 1
        
        rs = None
        
        if num_clusters > 0:
            clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
            noise = pd.Series([coords[cluster_labels==-1]])
            centermost_points = clusters.map(get_centermost_point)

            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'id':cur_id, x:lons, y:lats})
            result_labels = list(range(num_clusters))
            rs = pd.DataFrame({'id':cur_id, x:rep_points['lon'], y:rep_points['lat'], 'cluster': result_labels})

            ids_clusters_df = pd.concat([ids_clusters_df, rs], ignore_index=True)
            ids_clusters_df[['id', 'cluster']] = ids_clusters_df [['id', 'cluster']].astype(int)

        clusterised_locs = pd.concat([clusterised_locs, locs_cur_id], ignore_index=True)
        
        if cfg.measure_time:
            message = 'id id: {:,}\t{:,} points -> {:,} cluster(s); {:,.2f} s.'
            log.info(message.format(cur_id, len(locs_cur_id), len(rs), time.time()-start_time))
        
        if cfg.draw_map:
            if rs is not None:
                map_clusters = folium.FeatureGroup(name=f"clusters of id  {cur_id} ({len(rs)})", show=False)
                basic_map.add_child(map_clusters)
                for index, row in rs.iterrows():
                    lat, lon = float(row['lat']), float(row['lon'])
                    cluster_number = int(row['cluster'])
                    add2map(basic_map, [lat, lon],  int(cur_id), cluster_number=cluster_number, group=map_clusters,
                           is_cluster_point=True, points_of_cluster=len(clusters[cluster_number]))

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

    if cfg.measure_time:
        message = "running time: {:,.2f} s"
        log.info(message.format(time.time() - start_global_time))

    ids_clusters_df.to_csv(save_data_folder + f"/clusters_{launch_time}.csv", index=False);
    clusterised_locs.to_csv(save_data_folder + f"/data_with_clusters_{launch_time}.csv");

    if cfg.load_interests:
        ids_interests = interests[interests['id'].isin(id_LIST)].copy()
        ids_interests.index = ids_interests['id']
        ids_interests.drop(['id'], axis=1, inplace=True)
        ids_interests.sort_index(inplace=True)
        ids_interests.T.to_csv(save_data_folder + f"/selected_ids_interests_{launch_time}.csv");


if __name__ == "__main__":
    make_clusters()
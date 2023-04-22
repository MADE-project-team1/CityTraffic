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

def add2map(map, bs_pt, cell_nm, cluster=-1, cluster_point=False, color='red', group=None, points_of_cluster=None):
    if cluster_point:
        icon_obj = plugins.BeautifyIcon(
            icon='arrow-down', icon_shape='marker',
            border_color=color, text_color=color,
            number=cell_nm
        )
        popup_message = f"cluster: {cluster}\npoints: {points_of_cluster}"
        marker = Marker(location=[bs_pt[0], bs_pt[1]], tooltip='cluster:' + str(cluster),
                popup=popup_message, icon=icon_obj)
    else:
        marker = Marker(location=[bs_pt[0], bs_pt[1]], tooltip='cluster:' + str(cluster),
                popup='cluster: ' + str(cluster))

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

    df = pd.read_csv(cfg.locs_path)
    interests = pd.read_csv(cfg.interests_path)


    if isinstance(cfg.user_points_range, ListConfig):
        bounds = OmegaConf.to_container(cfg.user_points_range)
    else:
        if cfg.user_points_range >= 0:
            bounds = [0, cfg.user_points_range]
        else:
            bounds = [0, np.inf]

    unique_id = pd.unique(df['id'])
    users = unique_id[df['id'].value_counts().between(*bounds)]

    if cfg.dist_policy == 'const':
        MAX_DIST_M = cfg.max_dist

    if cfg.sample_policy == 'const':
        MIN_SAMPLES = cfg.min_samples

    if cfg.n_users == -1:
        USER_LIST = users
    elif isinstance(cfg.n_users, ListConfig):
        USER_LIST = OmegaConf.to_container(cfg.n_users)
    else:
        USER_LIST = rng.choice(users, size=cfg.n_users)

    epsilon = MAX_DIST_M / METERS_PER_RADIAN
    x = 'lon'
    y = 'lat'

    city_center = df[[y, x]].mean()

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

    users_clusters_df = pd.DataFrame({'id' : [], 'lat' : [], 'lon' : [], 'cluster' : []})
    new_data = pd.DataFrame(columns=df.columns)
    new_data['is_weekday'] = new_data['is_weekday'].astype(bool)
    new_data['cluster'] = pd.Series(dtype='int')

    for cur_user in USER_LIST:
        df1 = df[df['id'] == cur_user].copy()
        start_time = time.time()
        coords = df1[[y, x]].values 
        db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, **cfg.model_params).fit(np.radians(coords))
        cluster_labels = db.labels_
        df1['cluster'] = cluster_labels
        num_clusters = len(set(cluster_labels))
        
        clusters = pd.Series([coords[cluster_labels==n] for n in range(num_clusters)])
        noise = pd.Series([coords[cluster_labels==-1]])
        
        if clusters.shape[0] > 1:
            clusters = clusters[:-1]
            centermost_points = clusters.map(get_centermost_point)

            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'id':cur_user, x:lons, y:lats})
            result_labels = list(range(num_clusters - 1))
            rs = pd.DataFrame({'id':cur_user, x:rep_points['lon'], y:rep_points['lat'], 'cluster': result_labels})

        else:
            rep_points = df1[['lat', 'lon']].mean()
            rs = pd.DataFrame({'id':cur_user, x:[rep_points['lon']], y:[rep_points['lat']], 'cluster': [0]})
        
        users_clusters_df = pd.concat([users_clusters_df, rs], ignore_index=True)
        users_clusters_df [['id', 'cluster']] = users_clusters_df [['id', 'cluster']].astype(int)
        new_data = pd.concat([new_data, df1], ignore_index=True)
        if cfg.measure_time:
            message = 'user id: {:,}\t{:,} points -> {:,} cluster(s); {:,.2f} s.'
            log.info(message.format(cur_user, len(df1), len(rs), time.time()-start_time))
        
        if cfg.draw_map:
            map_clusters = folium.FeatureGroup(name=f"clusters of id  {cur_user} ({len(rs)})", show=False)
            basic_map.add_child(map_clusters)
            for index, row in rs.iterrows():
                lat, lon = float(row['lat']), float(row['lon'])
                cluster_number = int(row['cluster'])
                add2map(basic_map, [lat, lon],  int(cur_user), cluster=cluster_number, group=map_clusters,
                        cluster_point=True, points_of_cluster=len(clusters[cluster_number]))

            all_locations = folium.FeatureGroup(name=f"locs of id {cur_user} ({len(df[df['id'] == cur_user])})" , show=False)
            basic_map.add_child(all_locations)
            for index, row in df1.iterrows():
                lat, lon = float(row['lat']), float(row['lon'])
                cluster_number = int(row['cluster'])
                add2map(basic_map, [lat, lon], int(cur_user), cluster=cluster_number, group=all_locations)


    if cfg.draw_map:
        folium.LayerControl().add_to(basic_map);
        MousePosition().add_to(basic_map);
        basic_map.save(save_data_folder + f"/map_{launch_time}.html")
    if cfg.measure_time:
        message = "running time: {:,.2f} s"
        log.info(message.format(time.time() - start_global_time))

    users_clusters_df.to_csv(save_data_folder + f"/clusters_{launch_time}.csv", index=False);
    users_interests = interests[interests['id'].isin(USER_LIST)].copy()
    users_interests.index = users_interests['id']
    users_interests.drop(['id'], axis=1, inplace=True)
    users_interests.sort_index(inplace=True)
    users_interests.T.to_csv(save_data_folder + f"selected_users_interests_{launch_time}.csv");
    new_data.to_csv(save_data_folder + f"data_with_clusters_{launch_time}.csv");


if __name__ == "__main__":
    make_clusters()
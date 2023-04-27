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

from draw import draw_map

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def calc_summary_time(x):
    return x['last_ts'].sum() - x['first_ts'].sum()

def calc_daily_mean_time(x):
    gb = x.groupby('log_date')[['first_ts', 'last_ts']].sum()
    mu = (gb['last_ts'] - gb['first_ts']).mean()
    std = (gb['last_ts'] - gb['first_ts']).std()
    return (mu, std)


METERS_PER_RADIAN = 6371008.8
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def make_clusters(cfg: DictConfig):

    #initial setup and data loading

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
        ID_LIST = ids
    elif isinstance(cfg.n_ids, ListConfig):
        ID_LIST = OmegaConf.to_container(cfg.n_ids)
    else:
        ID_LIST = rng.choice(ids, size=cfg.n_ids)

    epsilon = MAX_DIST_M / METERS_PER_RADIAN
    x = 'lon'
    y = 'lat'

    city_center = cfg.city_center

    start_global_time = time.time()

    #ids_clusters_df: DataFrame of all cluster centers
    ids_clusters_df = pd.DataFrame({'id' : [], 'lat' : [], 'lon' : [], 'cluster' : []})

    #ids_clusters_df: initial DataFrame with cluster labels assigned to every loc point
    clusterised_locs = pd.DataFrame(columns=locs.columns)
    clusterised_locs['is_weekday'] = clusterised_locs['is_weekday'].astype(bool)
    clusterised_locs['cluster'] = pd.Series(dtype='int')

    for cur_id in ID_LIST:
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
            result_labels = [cluster_labels[cluster_labels==n][0] for n in range(num_clusters)]
            cluster_size = [len(cluster_labels[cluster_labels==n]) for n in range(num_clusters)]
            geohashes = [
                gh.encode(latlon_g[0], latlon_g[1], precision=cfg.gh_interests_prec, bits_per_char=cfg.gh_bits_per_char)
                for latlon_g in centermost_points]
            # [locs_cur_id[locs_cur_id['cluster'] == n] for n in range(num_clusters)]
            mu_std = [calc_daily_mean_time(locs_cur_id[locs_cur_id['cluster'] == n]) for n in range(num_clusters)]
            mean_daily_time = [int(mu_std[i][0]) for i in range(num_clusters)]
            std_daily_time = [int(mu_std[i][1]) for i in range(num_clusters)]

            rs = pd.DataFrame({'id':cur_id, 
                               x:rep_points['lon'], 
                               y:rep_points['lat'],
                               'cluster': result_labels, 
                               'cluster_size': cluster_size, 
                               'geohash': geohashes,
                               'mean_daily_time(s)': mean_daily_time,
                               'std_daily_time(s)': std_daily_time})


        elif pd.unique(cluster_labels) != -1:
            clusters = coords
            centermost_points = get_centermost_point(clusters)
            lats, lons = centermost_points
            rep_points = pd.DataFrame({'id':cur_id, x:lons, y:lats}, index=[0])
            result_labels = [0]
            cluster_size = len(coords)
            geohashes = [gh.encode(lats, lons, precision=cfg.gh_interests_prec, bits_per_char=cfg.gh_bits_per_char)]
            
            rs = pd.DataFrame({'id':cur_id, 
                                 x:rep_points['lon'],
                                 y:rep_points['lat'],
                                 'cluster': result_labels,
                                 'cluster_size': cluster_size, 
                                 'geohash':geohashes}, index=[0]
                            )
            
        ids_clusters_df = pd.concat([ids_clusters_df, rs], ignore_index=True)


        clusterised_locs = pd.concat([clusterised_locs, locs_cur_id], ignore_index=True)
        
        if cfg.measure_time:
            message = 'id id: {:,}\t{:,} points -> {:,} cluster(s); {:,.2f} s.'
            log.info(message.format(cur_id, len(locs_cur_id), len(rs), time.time()-start_time))

    ids_clusters_df[['id', 'cluster', 'cluster_size']] = ids_clusters_df [['id', 'cluster', 'cluster_size']].astype(int)
    ids_clusters_df[['geohash']] = ids_clusters_df[['geohash']].astype(str)

    if cfg.measure_time:
        message = "running time: {:,.2f} s"
        log.info(message.format(time.time() - start_global_time))

    ids_clusters_df.to_csv(save_data_folder + f"/clusters_{launch_time}.csv", index=False);
    clusterised_locs.to_csv(save_data_folder + f"/data_with_clusters_{launch_time}.csv");

    if cfg.load_interests:
        ids_interests = interests[interests['id'].isin(ID_LIST)].copy()
        ids_interests.index = ids_interests['id']
        ids_interests.drop(['id'], axis=1, inplace=True)
        ids_interests.sort_index(inplace=True)
        ids_interests.T.to_csv(save_data_folder + f"/selected_ids_interests_{launch_time}.csv");

    if cfg.draw_map:
        draw_map(save_data_folder, launch_time, cfg, log=log)



if __name__ == "__main__":
    make_clusters()
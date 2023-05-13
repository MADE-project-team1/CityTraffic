import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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

from draw import draw_map

from tqdm import tqdm

#gets cluster latlons and returns its center
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

METERS_PER_RADIAN = 6371008.8
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='config', config_name='config')
def make_clusters(cfg: DictConfig):

    #initial setup and data loading

    launch_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_data_folder = f"./data/{launch_time}"
    os.mkdir(save_data_folder)
    rng = np.random.default_rng(seed=cfg.seed)

    locs = pd.read_csv(cfg.cleaned_locs_path)

    if isinstance(cfg.id_points_range, ListConfig):
        bounds = OmegaConf.to_container(cfg.id_points_range)
    else:
        if cfg.id_points_range >= 0:
            bounds = [0, cfg.id_points_range]
        else:
            bounds = [0, np.inf]

    ids = pd.unique(locs['id'])

    if cfg.dist_policy == 'const':
        MAX_DIST_M = cfg.max_dist
    
    if cfg.only_hw:
        denom = cfg.min_sample_hw
    else:
        denom = cfg.min_sample_interest

    if cfg.n_ids == -1:
        ID_LIST = ids
    elif isinstance(cfg.n_ids, ListConfig):
        ID_LIST = OmegaConf.to_container(cfg.n_ids)
    else:
        ID_LIST = rng.choice(ids, size=cfg.n_ids)

    epsilon = MAX_DIST_M / METERS_PER_RADIAN

    #short naming
    x = 'lon'
    y = 'lat'
    
    columns=['lat', 'lon', 'ts', 'id', 'length']

    city_center = cfg.city_center

    start_global_time = time.time()

    #ids_clusters_df: DataFrame of all cluster centers
    ids_clusters_df = pd.DataFrame({'id' : [], 'lat' : [], 'lon' : [], 'cluster' : []})

    #clusterised_locs: initial DataFrame with cluster labels assigned to every loc point
    clusterised_locs = pd.DataFrame(columns=columns)
    clusterised_locs['cluster'] = pd.Series(dtype='int')

    for cur_id in tqdm(ID_LIST):
        cur_id_df = locs.query('id == @cur_id')
        if cur_id_df['cnt'].sum() > bounds[1]:
            continue
        new_id_df = np.empty((0, len(columns)))

        for index, row in cur_id_df.iterrows():
            cnt = row['cnt']
            first_ts = row['first_ts']
            last_ts = row['last_ts']
            lats =  np.ones((cnt, 1)) * row['lat']
            lons =  np.ones((cnt, 1)) * row['lon']
            ids = np.ones((cnt, 1)) * cur_id
            points = np.linspace(first_ts, last_ts, cnt).reshape(((cnt, 1)))
            lengths = np.ones((cnt, 1)) * ((last_ts - first_ts) / cnt)
            values = np.hstack([lats, lons, points, ids, lengths])
            new_id_df = np.vstack([new_id_df, values])

            
        new_id_df = pd.DataFrame(data=new_id_df,
                columns=columns)
        
        new_id_df['log_date'] = pd.to_datetime(new_id_df['ts'], unit='s').dt.strftime('%Y-%m-%d')
        
        coords = new_id_df[['lat', 'lon']].values
        
        MIN_SAMPLES = max(new_id_df.shape[0] // denom, 100)
        # log.info(f"{cur_id}: {coords.shape}")
        db = DBSCAN(eps=epsilon, min_samples=MIN_SAMPLES, **cfg.model_params).fit(np.radians(coords))
        new_id_df['cluster'] = db.labels_

        clusters_list = pd.unique(new_id_df['cluster'][new_id_df['cluster'] != -1])
        if len(clusters_list) > 0:
            clusters = pd.Series([coords[db.labels_== c] for c in clusters_list])
            centermost_points = clusters.map(get_centermost_point)
            lats, lons = zip(*centermost_points)
            rep_points = pd.DataFrame({'id':cur_id, x:lons, y:lats})
            result_labels = [db.labels_[db.labels_== c][0] for c in clusters_list]
            cluster_size = [len(db.labels_[db.labels_== c]) for c in clusters_list]

            rs = pd.DataFrame({'id':cur_id, 
                    x:rep_points['lon'], 
                    y:rep_points['lat'],
                    'cluster': result_labels, 
                    'cluster_size': cluster_size,
                    })

            ids_clusters_df = pd.concat([ids_clusters_df, rs], ignore_index=True)

        clusterised_locs = pd.concat([clusterised_locs, new_id_df], ignore_index=True)


    ids_clusters_df.to_csv(save_data_folder + f"/clusters_{launch_time}.csv", index=False);
    clusterised_locs.to_csv(save_data_folder + f"/data_with_clusters_{launch_time}.csv");

if __name__ == "__main__":
    make_clusters()
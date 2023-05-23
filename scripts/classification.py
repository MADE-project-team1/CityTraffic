import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='config', config_name='config')
def classify(cfg: DictConfig):
    cluster_data = pd.read_csv(f"data\\{cfg.launch_time}\\clusters_{cfg.launch_time}.csv")
    df = pd.read_csv(f"data\\{cfg.launch_time}\\data_with_clusters_{cfg.launch_time}.csv", index_col=0)

    INQ_THRESH = cfg.inq_thresh
    WD_ALL_THRESH = cfg.wd_all_thresh
    SEC_IN_DAY = 60 * 60 * 24

    #TODO: remove this block
    id_day_info = df.groupby('id')['log_date'].nunique()
    good_id = id_day_info[id_day_info > 10].index
    df = df[df['id'].isin(good_id)]
    cluster_data = cluster_data[cluster_data['id'].isin(good_id)]

    df['day_ts'] = df['ts'] % (SEC_IN_DAY)

    A = cfg.bottom_bound_day
    B = cfg.top_bound_day
    DAY_NIGHT_THRESH = cfg.day_night_thresh

    if 'work_place' in cluster_data.columns:
        cluster_data.drop(columns=['work_place'], inplace=True)
        
    cluster_data['inq'] = None
    cluster_data['wd_rate'] = None
    cluster_data['wd_all'] = None

    for cur_id in tqdm(df['id'].unique()):
        cur_id_df = df.query('id == @cur_id and cluster != -1').copy()

        #weekdays
        wd_data = cur_id_df[pd.to_datetime(cur_id_df['ts'], unit='s').dt.dayofweek < 5].copy()
        df_gd = wd_data.groupby('cluster')
        daytime = df_gd['day_ts'].apply(lambda x: np.sum(x.between(A, B)) / x.shape[0])
        day_clusters = daytime[daytime > DAY_NIGHT_THRESH].index
        night_clusters = daytime[daytime <= DAY_NIGHT_THRESH].index

        #day clusters
        day_gd = wd_data[wd_data['cluster'].isin(day_clusters)].groupby('cluster')
        d_inq = day_gd[['day_ts']].agg(lambda x: np.quantile(x, 0.75)) - day_gd[['day_ts']].agg(lambda x: np.quantile(x, 0.25))
        day_features = pd.DataFrame({'cluster' : day_clusters})
        day_features = day_features.merge(d_inq['day_ts'].rename('inq').reset_index(), on='cluster', how='left')


        #night clusters
        night_data = wd_data[~wd_data['cluster'].isin(day_clusters)]
        n_mask = night_data['day_ts'] < night_data['day_ts'].mean()
        night_data.loc[n_mask, 'day_ts'] = SEC_IN_DAY + night_data['day_ts'][n_mask] 
        night_gd = night_data.groupby('cluster')
        n_inq = night_gd[['day_ts']].agg(lambda x: np.quantile(x, 0.75)) - night_gd[['day_ts']].agg(lambda x: np.quantile(x, 0.25))
        night_features = pd.DataFrame({'cluster' : night_clusters})
        night_features = night_features.merge(n_inq['day_ts'].rename('inq').reset_index(), on='cluster', how='left')

        features = pd.concat([day_features, night_features])


        wdays_number = wd_data.groupby('cluster')['log_date'].nunique()
        wd_day_rate = wdays_number  / wd_data['log_date'].nunique()
        wd_day_rate = wd_day_rate.rename('wd_rate')
        features = features.merge(wd_day_rate, on='cluster')

        wd_all = wdays_number  / wdays_number.sum()
        wd_all = wd_all.rename('wd_all')
        features = features.merge(wd_all, on='cluster')

        mask = cluster_data['id'] == cur_id
        cluster_data.loc[mask] = cluster_data.loc[mask].drop(columns=['inq', 'wd_rate', 'wd_all']).merge(features, on='cluster', how='left').values
    cluster_data.dropna(inplace=True)

    cluster_data['work_place'] = (cluster_data['wd_all'] < WD_ALL_THRESH) | ((cluster_data['wd_all'] >= WD_ALL_THRESH) & (cluster_data['inq'] < INQ_THRESH))
    cluster_data = cluster_data[cluster_data['inq'].between(cfg.low_filter_time, cfg.top_filter_time)]
    cluster_data.to_csv(f'data/{cfg.launch_time}/labeled_cluster_data_{cfg.launch_time}.csv')

if __name__ == "__main__":
    classify()



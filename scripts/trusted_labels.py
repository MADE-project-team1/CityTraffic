import pandas as pd
import hydra
from omegaconf import DictConfig
import numpy as np

def map_slot(x, number_of_slots):
    if x > number_of_slots // 2:
        return number_of_slots - x
    return x

@hydra.main(version_base=None, config_path='config', config_name='config')
def labeling(data_path: str, cluster_data_path: str, save_path: str, cfg: DictConfig):
    ts = pd.read_csv(data_path, index_col=0)
    cluster_data = pd.read_csv(cluster_data_path)
    
    func = lambda x: map_slot(x, cfg.number_of_slots)

    ts['first_slot'] = ts['first_slot'].apply(func)
    ts['last_slot'] = ts['last_slot'].apply(func)
    cluster_data['mean_days'] = None
    cluster_data['activity'] = None
    cluster_data['mean_time'] = None
    cluster_data['label'] = 'no'

    id_test = pd.unique(ts['id'])

    weekday = True

    for ts_id in id_test:
        id_ts = ts.query('id == @ts_id and is_weekday == @weekday')
        total_days = len(pd.unique(id_ts['log_date']))
        for cur_cluster in pd.unique(id_ts['cluster']):
            cur_cluster_ts = id_ts.query('cluster == @cur_cluster')

            slots = cur_cluster_ts[['first_slot', 'last_slot']].values.flatten()
            number_of_days = len(pd.unique(cur_cluster_ts['log_date']))

            activity = len(slots) / number_of_days
            mean_days = number_of_days / total_days
            mean_time = slots.mean()

            #predicting
            
            cluster_data.loc[(cluster_data['id'] == ts_id) & (cluster_data['cluster'] == cur_cluster), ['mean_days', 'activity', 'mean_time']] = mean_days, activity, mean_time
    
    weekday = False
    cluster_data['mean_days_we'] = None
    cluster_data['activity_we'] = None
    cluster_data['mean_time_we'] = None

    for ts_id in id_test:
        id_ts = ts.query('id == @ts_id and is_weekday == @weekday')
        total_slots = len(np.unique(id_ts[['first_slot', 'last_slot']].values.flatten()))
        total_days = len(pd.unique(id_ts['log_date']))
        for cur_cluster in pd.unique(id_ts['cluster']):
            cur_cluster_ts = id_ts.query('cluster == @cur_cluster')

            slots = cur_cluster_ts[['first_slot', 'last_slot']].values.flatten()
            number_of_days = len(pd.unique(cur_cluster_ts['log_date']))
            activity = len(slots) / number_of_days
            mean_days = number_of_days / total_days
            mean_time = slots.mean()

            #predicting
            
            cluster_data.loc[(cluster_data['id'] == ts_id) & (cluster_data['cluster'] == cur_cluster), ['mean_days_we', 'activity_we', 'mean_time_we']] = mean_days, activity, mean_time
            
    MEAN_THRESH_T = 15
    MEAN_THRESH_B = 14
    DAYS_RATE_THRESH = 0.8
    ACTIVE_THRESH = 20

    def put_label(x):
        if x['mean_days'] >= DAYS_RATE_THRESH and x['activity'] >= ACTIVE_THRESH:
            if MEAN_THRESH_B  <= x['mean_time']:
                x['label'] = 'work'
            else:
                x['label'] = 'home'
        else:
            x['label'] = 'interest'
        
        return x
    

    id_test = pd.unique(ts['id'])
    labeled_data = cluster_data[cluster_data['id'].isin(id_test)].dropna(axis=0).apply(put_label, axis=1)
    labeled_data.to_csv(save_path)
    



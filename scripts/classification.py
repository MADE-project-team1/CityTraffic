import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path='config', config_name='config')
def classify(cfg: DictConfig):
    cluster_data = pd.read_csv(cfg.cluster_data_path)
    cluster_data['work_place'] = None

    df = pd.read_csv(cfg.cleaned_data_path)


    THRESH_VALUE = cfg.inq_thresh
    SEC_IN_DAY = 60 * 60 * 24

    #only weekday

    for cur_id in tqdm(df['id'].unique()):
        cur_id_df = df.query('id == @cur_id')
        data = cur_id_df[(cur_id_df['cluster'] != -1) & (pd.to_datetime(cur_id_df['ts'], unit='s').dt.dayofweek < 5)].copy()
        data['day_ts'] = cur_id_df['ts'] % (SEC_IN_DAY)
        gd = data.groupby('cluster')
        inq = gd[['day_ts']].agg(lambda x: np.quantile(x, 0.75)) - gd[['day_ts']].agg(lambda x: np.quantile(x, 0.25))
        inq = (inq < THRESH_VALUE)['day_ts'].rename('work_place').reset_index()
        cluster_data.loc[cluster_data['id'] == cur_id] = cluster_data.loc[cluster_data['id'] == cur_id].drop(columns='work_place').merge(inq, on='cluster', how='left').values
    
    cluster_data.fillna(True)


if __name__ == "__main__":
    classify



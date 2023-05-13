import pandas as pd
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import datetime
import os
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from tqdm import tqdm


def get_slot(x, slot_len):
    hm = pd.to_datetime(x, unit='s').dt.strftime('%H:%M').str.split(':')
    return hm.apply(lambda x: (60 * int(x[0]) + int(x[1])) // slot_len)

@hydra.main(version_base=None, config_path='config', config_name='config')
def clean(cfg: DictConfig):

    launch_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_data_folder = f"./data/{launch_time}"
    os.mkdir(save_data_folder)
    rng = np.random.default_rng(seed=cfg.seed)

    if isinstance(cfg.id_points_range, ListConfig):
        bounds = OmegaConf.to_container(cfg.id_points_range)
    else:
        if cfg.id_points_range >= 0:
            bounds = [0, cfg.id_points_range]
        else:
            bounds = [0, np.inf]

    locs = pd.read_csv(cfg.locs_path)

    if cfg.n_ids == -1:
        ID_LIST = pd.unique(locs['id'])
    elif isinstance(cfg.n_ids, ListConfig):
        ID_LIST = OmegaConf.to_container(cfg.n_ids)
    else:
        ID_LIST = rng.choice(pd.unique(locs['id']), size=cfg.n_ids)

    new_df = pd.DataFrame(columns=locs.columns)
    for cur_id in tqdm(ID_LIST):
            cur_id_df = locs.query('id == @cur_id').copy()
            if cur_id_df['cnt'].sum() > bounds[1] or cur_id_df['cnt'].sum() < bounds[0]:
                continue
            cur_id_df['length'] = cur_id_df['last_ts'] - cur_id_df['first_ts']
            days = cur_id_df.groupby('log_date')['length'].sum()
            days = days[days >= cfg.min_time_day_thresh]
            if days.shape[0] >= cfg.min_days_thresh:
                filtered_data = cur_id_df[cur_id_df['log_date'].isin(days.index)]
                new_df = pd.concat([new_df, filtered_data])
    
    new_df.to_csv(f'data/cleaned_locs_data_{cfg.min_time_day_thresh}_{cfg.min_days_thresh}.csv')

if __name__ == '__main__':
    clean()
        
    
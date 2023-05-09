import pandas as pd
import hydra
from omegaconf import DictConfig


def get_slot(x, slot_len):
    hm = pd.to_datetime(x, unit='s').dt.strftime('%H:%M').str.split(':')
    return hm.apply(lambda x: (60 * int(x[0]) + int(x[1])) // slot_len)

@hydra.main(version_base=None, config_path='config', config_name='config')
def clean(cluster_data_path: str, cfg: DictConfig):
    save_data_folder = f"./data"

    clean_df = pd.read_csv(cfg.locs_path)

    if cfg.only_weekday:
        clean_df = clean_df.query('is_weekday == True')

    if cfg.drop_noise:
        clean_df = clean_df.query('cluster != -1')
    
    new_df = pd.DataFrame(columns=clean_df.columns)

    id_list = pd.unique(clean_df['id'])

    for cur_id in id_list:
        cur_id_df = clean_df.query("id == @cur_id")
        func = lambda x: get_slot(x, cfg.slot_len)
        cur_id_df.loc[:, ['first_slot', 'last_slot']] = cur_id_df[['first_ts', 'last_ts']].apply(func).values

        days_info = cur_id_df[['log_date', 'first_slot', 'last_slot']].set_index('log_date').stack().groupby('log_date')\
            .agg(first_slot='unique', last_slot='unique')

        mapped = days_info['first_slot'].map(len)
        good_days = mapped[mapped >= cfg.min_slots].index
        
        if len(good_days) >= cfg.min_days:
            new_df = pd.concat([new_df, cur_id_df[cur_id_df['log_date'].isin(good_days)]], ignore_index=True)
    
    new_df.to_csv('cleaned_locs_data.csv')

    if cfg.cluster_data_path:
        cluster_data = pd.read_csv(cluster_data_path)
        cleaned_cluster_data = cluster_data[cluster_data['id'].isin(pd.unique(new_df['id']))]
        cleaned_cluster_data.to_csv(cfg.cluster_data_path)
        
    
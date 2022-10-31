import utils
from datetime import datetime
import os
import pandas as pd

from experiment_butter_buy import run_experiment

if __name__ == '__main__':
    args = utils.parse_config()
    utils.create_logger_and_set_level(args.verbose)

    # prep file to save results
    results_file_name = datetime.now().strftime("%Y%m%d_%H%M") + '_experiment_results'
    file_dir = os.path.join('results/', results_file_name)

    experiment_name = 'shelf_life'
    diff_shelf_lives = [1, 3, 8, 10, 13, 18, 23, 30, 40]

    for shelf_life in diff_shelf_lives:
        # define buyer properties
        properties = {
            'product_shelf_life': shelf_life,
            'ordering_cost': 0.1,
            'storage_capacity': 40000,
            'min_inventory_threshold': 3000,
            'consumption_rate': 3000,
            'storage_cost': 0.2,
            'cash_inflow': 6400000,  # ±3200 product
            'upper_buy_limit': 10000
        }

        df = pd.read_csv('./data/US_SMP_food_TA.csv', index_col=0).iloc[69:].reset_index(drop=True).sort_values('ds')
        train_fraction = .75
        train_df = df.iloc[:round(train_fraction * len(df))]
        test_df = df.iloc[round(train_fraction * len(df)):].reset_index(drop=True)

        ts_feature_names = \
            ["y", "y_24_quo", "y_26_quo", "y_37_quo", "y_94_quo", "y_20_quo", "y_6_quo", "y_227_pro", "y_785_end",
             "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band", "momentum1",
             "kalman"]

        run_experiment(args, properties, train_df, test_df, ts_feature_names,
                       file_dir+f'_{experiment_name}_{shelf_life}.csv')


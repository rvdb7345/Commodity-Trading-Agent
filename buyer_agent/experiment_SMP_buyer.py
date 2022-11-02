"""This file is used for comparing the agent's performance against the baseline accounting for stochasisticity
through repetitions.


Created by Vesper in cooperation with Slimmmer AI.
https://www.vespertool.com/
"""

from datetime import datetime
import os
import pandas as pd
import utils

from agent_SMP_buyer import train_and_simulate
from scipy.stats import wilcoxon
from tqdm import tqdm


def run_experiment(args, properties, train_df, test_df, ts_feature_names, file_dir):
    """Train and evaluate the agent in agent_SMP_buyer.py a set number times and test whether it is significantly
    better than the baseline."""

    experiment_results = pd.DataFrame(
        columns=['total_worth_model_train', 'total_worth_baseline_train', 'total_worth_model_test',
                 'total_worth_baseline_test'], index=list(range(args.reps)))
    for rep_id in tqdm(range(args.reps)):
        results_dict_train, \
        results_dict_test, \
        results_dict_baseline_train, \
        results_dict_baseline_test = train_and_simulate(args, train_df, test_df, ts_feature_names, properties,
                                                        verbose=False)

        experiment_results.loc[rep_id, :] = [results_dict_train['total_worth'],
                                             results_dict_baseline_train['total_worth'], \
                                             results_dict_test['total_worth'],
                                             results_dict_baseline_test['total_worth']]

        # save intermediate results to not lose progress
        experiment_results.to_csv(file_dir, mode='w', header=True, index=False)

    for dataset in ['train', 'test']:
        print(f'\nEvaluating for {dataset} set')
        experiment_results[f'improvement_{dataset}'] = (experiment_results[f'total_worth_model_{dataset}'] -
                                                        experiment_results[f'total_worth_baseline_{dataset}']) / \
                                                       experiment_results[f'total_worth_baseline_{dataset}'] * 100

        print(f"Experiment improvement over baseline mean: {experiment_results[f'improvement_{dataset}'].mean():.2f}%")
        print(f"Experiment improvement over baseline std: {experiment_results[f'improvement_{dataset}'].std():.2f}%")

        w, p = wilcoxon(experiment_results[f'improvement_{dataset}'], alternative='greater')

        # perform wilcoxon on %improvement over baseline
        print(f"\nWilcoxon test on the {dataset} results")
        print(f"H0: model performance = baseline performance")
        print(f"H1: model performance > baseline performance")
        print(f"Test p-value: {p:.12f}%")
        if p <= 0.05:
            print("H0 is rejected, model performance is better than the baseline")
        else:
            print("H0 cannot be rejected, the model is NOT significantly better than the baseline")

    print(experiment_results.round(2))

    return experiment_results


if __name__ == '__main__':
    args = utils.parse_config()
    utils.create_logger_and_set_level(args.verbose)

    # prep file to save results
    results_file_name = datetime.now().strftime("%Y%m%d_%H%M") + '_experiment_results.csv'
    file_dir = os.path.join('../results/', results_file_name)

    # define buyer properties
    properties = {
        'product_shelf_life': 13,
        'ordering_cost': 0.1,
        'storage_capacity': 40000,
        'min_inventory_threshold': 3000,
        'consumption_rate': 3000,
        'storage_cost': 0.2,
        'cash_inflow': 6400000,  # ±3200 product
        'upper_buy_limit': 10000
    }

    df = pd.read_csv('../data/US_SMP_food_TA.csv', index_col=0).iloc[69:].reset_index(drop=True).sort_values('ds')
    train_fraction = .75
    train_df = df.iloc[:round(train_fraction * len(df))]
    test_df = df.iloc[round(train_fraction * len(df)):].reset_index(drop=True)

    ts_feature_names = \
        ["y", "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band", "momentum1",
         "kalman"]

    run_experiment(args, properties, train_df, test_df, ts_feature_names, file_dir)

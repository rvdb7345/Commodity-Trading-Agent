import pandas as pd
import utils

from agent_butter_buyer import train_and_simulate
from scipy.stats import wilcoxon
from tqdm import tqdm

if __name__ == '__main__':
    args = utils.parse_config()
    utils.create_logger_and_set_level(args.verbose)

    df = pd.read_csv('./data/US_SMP_food_TA.csv', index_col=0).iloc[69:].reset_index(drop=True).sort_values('ds')
    ts_feature_names = \
        ["y", "y_24_quo", "y_26_quo", "y_37_quo", "y_94_quo", "y_20_quo", "y_6_quo", "y_227_pro", "y_785_end",
        "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band", "momentum1", "kalman"]
        
    # define buyer properties
    properties = {
        'product_shelf_life': 13,
        'ordering_cost': 0.1,
        'storage_capacity': 40000,
        'min_inventory_threshold': 3000,
        'consumption_rate': 3000,
        'storage_cost': 0.2,
        'cash_inflow': 8000000,
        'upper_buy_limit': 10000
    }

    train_fraction = .75
    train_df = df.iloc[:round(train_fraction*len(df))]
    test_df = df.iloc[round(train_fraction*len(df)):].reset_index(drop=True)

    experiment_results = pd.DataFrame(columns=['model_results', 'baseline_results'], index=list(range(args.reps)))
    for rep_id in tqdm(range(args.reps)):
        results_dict, results_dict_baseline = train_and_simulate(args, train_df, test_df, ts_feature_names, properties, verbose=False)

        experiment_results.loc[rep_id,:] = [results_dict['total_worth'], results_dict_baseline['total_worth']]
        print(f'\nExperiment {rep_id}')
        # print(f"Total worth (incl inventory) agent butter: {results_dict['total_worth']:.2f}")
        # print(f"Total worth (incl inventory) baseline: {results_dict_baseline['total_worth']:.2f}")

        # print(f"Total worth improvement over baseline: {(results_dict['total_worth']-results_dict_baseline['total_worth'])/results_dict_baseline['total_worth']*100:.4f}%")

    experiment_results['improvement'] = (experiment_results['model_results']-experiment_results['baseline_results'])/experiment_results['baseline_results']*100
    print(experiment_results.head(args.reps))
    
    print(f"\nExperiment improvement over baseline mean: {experiment_results['improvement'].mean()}")
    print(f"Experiment improvement over baseline std: {experiment_results['improvement'].std()}")
    
    w, p = wilcoxon(experiment_results['improvement'], alternative='greater')
    
    # perform wilcoxon on %improvement over baseline
    print("\nWilcoxon test")
    print(f"H0: model performance = baseline performance")
    print(f"H1: model performance > baseline performance")
    print(f"Test p-value: {p}")
    if p <= 0.05:
        print("H0 is rejected, model performance is better than the baseline")
    else:
        print("H0 cannot be rejected, the model is NOT significantly better than the baseline")
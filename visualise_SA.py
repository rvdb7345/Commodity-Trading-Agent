import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

if __name__ == '__main__':

    exp_values = {
        'shelf_life': [3, 8, 13, 18]
    }

    param = 'shelf_life'

    results = []
    for value in exp_values[param]:
        experiment_results_df = pd.read_csv(f'results/20221019_1627_experiment_results_{param}_{value}.csv')
        experiment_results_df[param] = value
        results.append(experiment_results_df)

    total_results = pd.concat(results)

    total_results.dropna(inplace=True)

    total_results['test_improvement'] = total_results['total_worth_model_test'] / total_results['total_worth_baseline_test']
    total_results['train_improvement'] = total_results['total_worth_model_train'] / total_results['total_worth_baseline_train']

    SA_viz_results = total_results.groupby(param).agg(['mean', 'std'])

    SA_viz_results.reset_index(inplace=True)

    fig, ax = plt.subplots()

    # plt.plot(SA_viz_results['test_improvement'].index.values, SA_viz_results['test_improvement']['mean'])
    for type in ['train', 'test']:
        plt.figure()
        plt.title(f'Average Improvement over Baseline {type}')
        plt.ylabel('Improvement (%)')
        plt.xlabel(param)
        plt.errorbar(exp_values[param], SA_viz_results[f'{type}_improvement']['mean'], yerr=SA_viz_results[f'{type}_improvement']['std'], capsize=10)
        plt.show()
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

VESPER_cmap = np.loadtxt('colormaps/vesper_color_map.txt')
cm = colors.ListedColormap(VESPER_cmap / 255.0)

if __name__ == '__main__':

    exp_values = {
        'shelf_life': [1, 3, 8, 13, 18, 23, 30, 40, 52]
    }

    param = 'shelf_life'
    param_pretty = 'Shelf Life (weeks)'

    results = []
    for value in exp_values[param]:
        experiment_results_df = pd.read_csv(f'results/20221031_1510_experiment_results_{param}_{value}.csv')
        experiment_results_df[param] = value
        results.append(experiment_results_df)

    total_results = pd.concat(results)

    total_results.dropna(inplace=True)

    total_results['test_improvement'] = total_results['total_worth_model_test'] / total_results[
        'total_worth_baseline_test'] * 100
    total_results['train_improvement'] = total_results['total_worth_model_train'] / total_results[
        'total_worth_baseline_train'] * 100

    SA_viz_results = total_results.groupby(param).agg(['mean', 'std'])

    SA_viz_results.reset_index(inplace=True)

    fig, ax = plt.subplots()
    print(VESPER_cmap[0])

    # plt.plot(SA_viz_results['test_improvement'].index.values, SA_viz_results['test_improvement']['mean'])
    for type in ['train', 'test']:
        f, ax = plt.subplots(figsize=(10, 6))
        plt.title(f'Improvement over Baseline: {type} set')
        plt.ylabel('Improvement (%)')
        plt.xlabel(param_pretty)
        plt.fill_between(exp_values[param], SA_viz_results[f'{type}_improvement']['mean'] - SA_viz_results[f'{type}_improvement']['std'],
                         SA_viz_results[f'{type}_improvement']['mean'] + SA_viz_results[f'{type}_improvement']['std'],
                         alpha=0.8, color='#33CDC6')
        plt.plot(exp_values[param], SA_viz_results[f'{type}_improvement']['mean'], color="#030C38", marker='o')
        plt.xlim([min(exp_values[param])-1, max(exp_values[param])+1])
        # plt.grid()
        # ax.xaxis.grid(True)
        ax.set_facecolor('#EEF2F7')
        plt.rcParams['svg.fonttype'] = 'none'
        plt.savefig(f'figures/SA_{param}.svg')
        plt.show()
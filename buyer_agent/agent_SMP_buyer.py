"""File containing the Reinforcement Learning based Buyer Agent.

Created by Vesper in cooperation with Slimmmer AI.
https://www.vespertool.com/
"""
import os
from typing import Union
import gym
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym import spaces
import utils

logger = logging.getLogger('logger')

MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000000

VESPER_cmap = np.loadtxt('../colormaps/vesper_color_map.txt')
cm = colors.ListedColormap(VESPER_cmap / 255.0)


class BuyerEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, args, df: pd.DataFrame, properties: dict, ts_feature_names: list, save_results=False):
        """Initiate buyer environment, creating the observation and action spaces.

        Args:
            df: time series data
            properties: values dictating the properties of the buyer environment
            ts_feature_names: names of the time series features to use
        """
        super(BuyerEnvironment, self).__init__()
        self.df_y = df[['y']]
        self.df = self.data_scaler(df)
        self.ts_feature_names = ts_feature_names
        self.properties = properties

        # Actions of the format Buy x%, or refrain, etc.
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))

        self.lookback_period = 26
        self.observation_space = spaces.Dict({
            'time_series': spaces.Box(low=0, high=np.inf, shape=(len(ts_feature_names), self.lookback_period)),
            'env_props': spaces.Box(low=0, high=1, shape=(5,))
        })

        self.product_shelf_life = properties['product_shelf_life']
        self.ordering_cost = properties['ordering_cost']
        self.storage_capacity = properties['storage_capacity']
        self.min_inventory_threshold = properties['min_inventory_threshold']
        self.consumption_rate = properties['consumption_rate']
        self.storage_cost = properties['storage_cost']
        self.cash_inflow = properties['cash_inflow']
        self.counter = 0
        self.reward = []
        self.save_results = save_results

        # parameters
        self.upper_buy_limit = properties['upper_buy_limit']
        self.price_diff_scaler = 10  # price difference on range [0, 50]

    def data_scaler(self, df):
        """Function to scale the time series data for input into the agent."""
        scale_columns = np.setdiff1d(df.columns.values, ['ds'])
        df_to_scale = df[scale_columns]
        df_scaled = (df_to_scale - df_to_scale.min()) / (df_to_scale.max() - df_to_scale.min()) * 2 - 1
        df_scaled.insert(0, 'ds', df.ds)
        return df_scaled

    def reset_values(self):
        """Reset simulation values to starting values."""
        self.balance = INITIAL_ACCOUNT_BALANCE

        self.total_spent_value = 0
        self.counter = 0
        self.current_inventory = pd.DataFrame(columns=['time_in_storage', 'amount'], dtype=float)
        self.current_inventory.loc[self.counter] = [0, self.min_inventory_threshold * 6]

        if self.save_results:
            self.reward_tracker = np.zeros((100000, 2))
            self.buy_tracker = np.zeros((100000, 2))
            self.inventory_tracker = np.zeros((100000, 2))

    def reset(self) -> dict:
        """Reset the state of the environment to an initial state.

        Returns:
            Initial observation to start with
        """
        self.reset_values()

        # Set the current step to a random point within the data frame
        self.current_step = 0
        self.start_step = self.current_step

        return self._next_observation()

    def baseline_reset(self) -> dict:
        """Prepare class for baseline simulation."""
        self.current_step = self.start_step
        self.reset_values()

    def enable_simulation_dataset(self, test_df):
        """Prepare the class for the test set."""
        self.df_y = test_df[['y']]
        self.df = self.data_scaler(test_df)

    def _next_observation(self) -> dict:
        """Create observation dictionary with all features.

        Returns:
            Next inputs for model
        """
        current_price = self.df_y.loc[self.current_step, "y"]
        self.cash_buy_limit = int(self.balance / current_price)
        self.storage_buy_limit = self.storage_capacity - self.current_inventory['amount'].sum()

        self.min_buy_need = self.min_inventory_threshold - self.current_inventory['amount'].sum()
        self.min_buy_need = max([0, self.min_buy_need])

        # Get the data points for the last 'look_back' weeks and scale to between 0-1
        frame = {
            'time_series': [
                self.df.iloc[self.current_step: self.current_step + self.lookback_period, :][feat_name].values
                for feat_name in self.ts_feature_names
            ],
            'env_props': [
                self.balance / 100000000000,
                self.current_inventory['amount'].sum() / self.storage_capacity,
                min([self.cash_buy_limit / self.storage_capacity, 1]),
                self.storage_buy_limit / self.storage_capacity,
                self.min_buy_need / self.min_inventory_threshold
            ]}

        logger.debug(f"Env variables [balance, inventory, cash_buy_lim, storage_buy_lim]")
        logger.debug(f"scaled values {['%.5f' % value for value in frame['env_props']]}")
        return frame

    def update_measure_tracker(self, action, reward):
        """Update measure tracker for plotting end results."""
        # if trackers too small, add rows
        if self.counter == self.buy_tracker.shape[1]:
            self.buy_tracker = np.vstack([self.buy_tracker, np.zeros(self.buy_tracker.shape)])
        if self.counter == self.reward_tracker.shape[1]:
            self.reward_tracker = np.vstack([self.reward_tracker, np.zeros(self.reward_tracker.shape)])
        if self.counter == self.inventory_tracker.shape[1]:
            self.inventory_tracker = np.vstack([self.inventory_tracker, np.zeros(self.inventory_tracker.shape)])

        # track the buys, rewards, inventory at every step
        self.buy_tracker[self.counter] = np.array([self.current_step, action])
        self.reward_tracker[self.counter] = np.array([self.current_step, reward])
        self.inventory_tracker[self.counter] = np.array([self.current_step, self.current_inventory['amount'].sum()])

    def step(self, action: float) -> Union[dict, float, bool, dict]:
        """Take the next action and update state and rewards.

        Composition of rewards function:

        reward
        - buy at the right price
        punish
        - missed opportunity
        - spoilage
        - inventory below threshold
        - action higher than storage/cash limit
        """
        logger.debug("---------------")
        logger.debug(f'Step: {self.current_step}')

        # multiply action in range [0,1] by upper_buy_limit factor (works better with NN model)
        action_buy_amount = action[0] * self.upper_buy_limit
        self._take_action(action_buy_amount)

        # make sure the current step does not move past the end of the ts
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'y'].values) - self.lookback_period:
            self.current_step = 0

        inv_under_min_reward = 0
        action_over_limit_reward = 0

        # reward buying at the correct price point
        current_price = self.df_y.loc[self.current_step, "y"]
        min_price_next_weeks = self.df_y.loc[self.current_step + 1:self.current_step + self.product_shelf_life,
                               "y"].min()
        price_profit = min_price_next_weeks - current_price
        buy_priceprofit_reward = price_profit * self.product_bought / self.upper_buy_limit / self.price_diff_scaler

        # punishment for missed price opportunity  
        buy_amount_weight = ((self.consumption_rate - self.product_bought) / self.upper_buy_limit)
        missed_opportunity_reward = price_profit / (
                    1 + self.product_bought / self.upper_buy_limit) / self.price_diff_scaler * buy_amount_weight

        # punishment for spoilage + update inventory
        sum_spoiled_product = self.current_inventory.loc[self.current_inventory['time_in_storage'] >
                                                         self.product_shelf_life, 'amount'].sum()
        spoil_reward = sum_spoiled_product / 1000

        # remove spoiled product
        self.current_inventory = self.current_inventory.loc[self.current_inventory['time_in_storage'] <=
                                                            self.product_shelf_life]

        # punishment for emergency buy to reach inventory threshold
        if self.min_buy_need > action_buy_amount:
            inv_under_min_reward = 3
            logger.debug(f'Emergency buy reward (to reach min inventory): -{inv_under_min_reward}')

        # punishment for buy action too large for storage or cash limits
        if action_buy_amount > self.cash_buy_limit or action_buy_amount > self.storage_buy_limit:
            action_over_limit_reward = 2
            logger.debug('Action over buy/storage limits: -1')

        # calculate total reward
        reward = buy_priceprofit_reward - missed_opportunity_reward - spoil_reward - \
                 inv_under_min_reward - action_over_limit_reward

        if self.save_results:
            self.update_measure_tracker(action_buy_amount, reward)

        # generate next observation
        obs = self._next_observation()

        # stop the simulation if we have negative balance
        done = False
        if self.balance < 0:
            done = True

        # increase the time in inventory
        self.current_inventory['time_in_storage'] = self.current_inventory['time_in_storage'] + 1
        self.counter += 1

        logger.debug(f'current price: {current_price}')
        logger.debug(f'Min price coming {self.product_shelf_life} weeks: {min_price_next_weeks}')
        logger.debug(f'The price profit reward: {buy_priceprofit_reward}')
        logger.debug(f'The missed opportunity reward: {-missed_opportunity_reward}')
        logger.debug(f'The spoiled product reward: {-spoil_reward}')
        logger.debug(f'Reward {reward}')
        logger.debug(f'New inventory: {self.current_inventory["amount"].sum()}')

        return obs, reward, done, {}

    def _take_action(self, action_buy_amount: float):
        """Update state parameters based on the action."""

        # Set the current price to a random price within the time step
        current_price = self.df_y.loc[self.current_step, "y"]

        logger.debug(f'Action {action_buy_amount}')

        # determine max and min amount of buyable product (storage and cash constraints, inventory threshold constraint)
        self.product_bought_upper_constr = min([self.storage_buy_limit, self.cash_buy_limit, action_buy_amount])
        self.product_bought = max([self.min_buy_need, self.product_bought_upper_constr])
        logger.debug(f'Product bought {self.product_bought}')

        # calculate cost of current acquisition
        additional_cost = self.product_bought * (current_price + self.ordering_cost)
        logger.debug(f'Additional cost {additional_cost}')

        # update balance
        self.balance -= additional_cost

        # update inventory
        if not self.product_bought == 0:
            self.current_inventory.loc[self.counter + 1] = [0, self.product_bought]

        self.total_spent_value += additional_cost

        # update inventory with spoiled and used product --> FIFO strategy
        if self.current_inventory['amount'].sum() > self.consumption_rate:
            to_be_consumed = self.consumption_rate

            while to_be_consumed != 0:
                idx_oldest_products = self.current_inventory['time_in_storage'].idxmax()
                oldest_product_amount = self.current_inventory.loc[idx_oldest_products, 'amount']

                if oldest_product_amount > to_be_consumed:
                    self.current_inventory.loc[idx_oldest_products, 'amount'] -= to_be_consumed
                    to_be_consumed = 0
                else:
                    self.current_inventory.drop(index=idx_oldest_products, inplace=True)
                    to_be_consumed -= oldest_product_amount
        else:
            self.current_inventory.drop(index=self.current_inventory.index, inplace=True)

        # update balance with operational costs
        self.balance += self.cash_inflow - self.storage_cost * self.current_inventory['amount'].sum()

        logger.debug(f'Balance: {self.balance}')

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""
        logger.info(f'Step: {self.current_step}')
        logger.info(f'Balance: {self.balance}')
        logger.info(f'product held: {self.current_inventory["amount"].sum()}')
        logger.info(f'Total spent value: {self.total_spent_value}')

    def return_results(self):
        """Return end results of a simulation."""
        current_inventory = self.current_inventory["amount"].sum()

        results_dict = {}
        results_dict['current_inventory'] = current_inventory
        results_dict['total_worth'] = self.balance + current_inventory * self.df_y.loc[self.current_step, "y"]
        results_dict['balance'] = self.balance
        results_dict['total_spent_value'] = self.total_spent_value
        return results_dict

    def plot_measure(self, measure='buys', dataset='test', fig_location=''):
        """Plot the behaviour of the buyer agent."""
        if measure.lower() == 'buys':
            measure_per_step = self.buy_tracker
            cmap = 'copper'
            norm = None
        if measure.lower() == 'reward':
            measure_per_step = self.reward_tracker
            cmap = 'RdYlGn'
            norm = colors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
        if measure.lower() == 'inventory':
            measure_per_step = self.inventory_tracker
            cmap = 'RdYlGn'
            norm = None

        # cut off all trailing zeros
        measure_per_step = measure_per_step[:self.counter]

        f, ax = plt.subplots(figsize=(10, 6))
        plt.title(f'{measure} per time step ({dataset})')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        plt.plot(measure_per_step[:, 0], self.df_y.loc[measure_per_step[:, 0], 'y'], linewidth=0.1)
        points = ax.scatter(measure_per_step[:, 0], self.df_y.loc[measure_per_step[:, 0], 'y'],
                            marker='o', c=measure_per_step[:, 1], cmap=cm, norm=norm)
        f.colorbar(points)
        ax.set_facecolor('#D0D8E6')
        plt.rcParams['svg.fonttype'] = 'none'
        plt.tight_layout()
        plt.savefig(f'{fig_location}/{measure}_colour_{dataset}.svg', dpi=300)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        plt.title(f'{measure} per time step ({dataset})')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        lns1 = ax.plot(measure_per_step[:, 0], self.df_y.loc[measure_per_step[:, 0], 'y'], linewidth=1, label='price',
                       color='black')
        ax2 = ax.twinx()
        lns2 = ax2.plot(measure_per_step[:, 0], measure_per_step[:, 1], linewidth=1, label=measure, color='red',
                        alpha=0.5)
        ax.set_facecolor('#D0D8E6')
        plt.rcParams['svg.fonttype'] = 'none'
        ax2.set_ylabel(f'{measure}')
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs)
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{fig_location}/{measure}_lines_{dataset}.svg', dpi=300)

    def set_saving(self, saving_mode=False):
        """Turn saving on or off."""
        self.save_results = saving_mode


def run_simulation(env, model, df, dataset, simsteps, plot=False, fig_location=''):
    """Run simulation of trained model."""
    if plot:
        env.env_method('set_saving', saving_mode=True)

    env.env_method('enable_simulation_dataset', df)
    obs = env.reset()

    for i in range(simsteps - 26):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i % 200 == 0:
            env.render()

    # show results
    logger.info("Model final results:")
    env.render()

    if plot:
        utils.plot_results(env=env, dataset=dataset, fig_location=fig_location)
    return env


def run_baseline_simulation(env, action, steps=1000):
    """Run baseline simulation."""
    # logging.getLogger('logger').setLevel(logging.INFO)

    obs = env.env_method('baseline_reset')
    for i in range(steps):
        _ = env.step([[action]])

    logger.info("Baseline final results")
    env.render()
    return env


def train_and_simulate(args, train_df, test_df, ts_feature_names, properties, verbose=20, fig_location=''):
    # setup vectorized env and model
    env = DummyVecEnv([lambda: BuyerEnvironment(args, train_df, properties, ts_feature_names)])
    model = PPO('MultiInputPolicy', env, verbose=verbose, learning_rate=0.01)

    # train model
    utils.run_and_track_runtime(model.learn, total_timesteps=args.trainsteps)

    # carry out simulation with train_set + run baseline
    env = run_simulation(env, model, train_df, 'train', simsteps=len(train_df), plot=args.plot, fig_location=fig_location)
    results_dict_train = env.env_method('return_results')[0]
    env = run_baseline_simulation(env=env, action=properties['consumption_rate'] / properties['upper_buy_limit'],
                                  steps=len(train_df))
    results_dict_baseline_train = env.env_method('return_results')[0]

    # carry out simulation with test_set + run baseline
    env = run_simulation(env, model, test_df, 'test', simsteps=len(test_df), plot=args.plot, fig_location=fig_location)
    results_dict_test = env.env_method('return_results')[0]
    env = run_baseline_simulation(env=env, action=properties['consumption_rate'] / properties['upper_buy_limit'],
                                  steps=len(test_df))
    results_dict_baseline_test = env.env_method('return_results')[0]

    return results_dict_train, results_dict_test, results_dict_baseline_train, results_dict_baseline_test


if __name__ == '__main__':
    args = utils.parse_config()
    utils.create_logger_and_set_level(args.verbose)

    # create time stamped folder for the figures
    if args.plot:
        # create a figure directory if necessary
        if not os.path.exists('../figures'):
            os.makedirs('../figures')

        figure_folder = datetime.now().strftime("%Y%m%d_%H%M")
        fig_save_location = "../figures/" + figure_folder
        os.mkdir(fig_save_location)
    else:
        fig_save_location = ''

    # define buyer properties
    properties = {
        'product_shelf_life': 52,
        'ordering_cost': 0.1,
        'storage_capacity': 3000 * 52,
        'min_inventory_threshold': 3000,
        'consumption_rate': 3000,
        'storage_cost': 0.2,
        'cash_inflow': 6400000,  # ±3200 product
        'upper_buy_limit': 10000
    }

    df = pd.read_csv('../data/US_SMP_food_TA.csv', index_col=0).reset_index(drop=True).sort_values('ds')
    ts_feature_names = ["y", "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band",
                        "momentum1", "kalman"]

    train_fraction = .75
    train_df = df.iloc[:round(train_fraction * len(df))]
    test_df = df.iloc[round(train_fraction * len(df)):].reset_index(drop=True)

    results_dict_train, results_dict_test, results_dict_baseline_train, results_dict_baseline_test = train_and_simulate(
        args, train_df, test_df, ts_feature_names, properties, fig_location=fig_save_location)

    print(f"Total worth (incl inventory) agent butter (train): {results_dict_train['total_worth']:.2f}")
    print(f"Total worth (incl inventory) baseline (train): {results_dict_baseline_train['total_worth']:.2f}")
    print(f"Total worth improvement over baseline (train): {(results_dict_train['total_worth'] - results_dict_baseline_train['total_worth']) / results_dict_baseline_train['total_worth'] * 100:.4f}%")

    print(f"Total worth (incl inventory) agent butter (test): {results_dict_test['total_worth']:.2f}")
    print(f"Total worth (incl inventory) baseline (test): {results_dict_baseline_test['total_worth']:.2f}")
    print(
        f"Total worth improvement over baseline (test): {(results_dict_test['total_worth'] - results_dict_baseline_test['total_worth']) / results_dict_baseline_test['total_worth'] * 100:.4f}%")
    plt.show()

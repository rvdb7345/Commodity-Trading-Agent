"""File containing the first draft of the Buyer Agent.
Created for the R&D C7 - RL trading agent project.
"""
import time
from typing import Union
import gym
import random
import logging

import numpy as np
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt, colors

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym import spaces
from gym.wrappers import FlattenObservation
from sb3_contrib import RecurrentPPO

import utils

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_PRODUCT = 2147483647
MAX_PRODUCT_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000000

logger = logging.getLogger('logger')

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
        self.df = df
        self.ts_feature_names = ts_feature_names
        self.properties = properties
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, or refrain, etc.
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))

        self.lookback_period = 26
        # self.observation_space = spaces.Box(low=0, high=1, shape=(21, self.lookback_period + 1), dtype=np.float16)
        self.observation_space = spaces.Dict({
            'time_series': spaces.Box(low=0, high=np.inf, shape=(20, self.lookback_period)),
            'env_props': spaces.Box(low=0, high=1, shape=(6,))
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
        self.action_scaler = 10000
        self.price_diff_scaler = 10


    def reset(self) -> dict:
        """Reset the state of the environment to an initial state.

        Returns:
            Initial observation to start with
        """
        self.balance = INITIAL_ACCOUNT_BALANCE

        self.cost_basis = 0
        self.total_spent_value = 0
        self.counter = 0
        self.current_inventory = pd.DataFrame(columns=['time_in_storage', 'amount'], dtype=float)
        self.current_inventory.loc[self.counter] = [0, self.min_inventory_threshold*6]

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'y'].values) - (self.lookback_period + 1))

        if self.save_results:
            self.reward_tracker = np.zeros((100000, 2))
            self.buy_tracker = np.zeros((100000, 2))
            self.inventory_tracker = np.zeros((100000, 2))

        return self._next_observation()

    def _next_observation(self) -> dict:
        """Create observation dictionary with all features.

        Returns:
            Next inputs for model
        """

        current_price = self.df.loc[self.current_step, "y"]
        self.cash_buy_limit = int(self.balance / current_price)
        self.storage_buy_limit = self.storage_capacity - self.current_inventory['amount'].sum()

        # Get the data points for the last 'look_back' weeks and scale to between 0-1
        frame = {
            'time_series': [
                self.df.iloc[self.current_step: self.current_step + self.lookback_period, :][feat_name].values
                for feat_name in self.ts_feature_names
            ],
            'env_props': [
                self.balance,
                self.current_inventory['amount'].sum()/self.storage_capacity,
                self.cost_basis,
                self.min_inventory_threshold,
                self.cash_buy_limit,
                self.storage_buy_limit,                
        ]}

        return frame

    def step(self, action: float) -> Union[dict, float, bool, dict]:
        """Take the next action and update observations and rewards."""
        logger.debug("---------------")
        # Execute one time step within the environment
        logger.debug(f'Current step: {self.current_step}')

        # multiply action by 10.000 because of NN output constraints
        action = action[0]*self.action_scaler
        self._take_action(action)

        # make sure the current step does not move past the end of the ts
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'y'].values) - self.lookback_period:
            self.current_step = 0
        
        ### calculate reward
        '''
        reward
        - as cheap as possible
        punish
        - inventory below threshold
        - inventory below 0 (high punishment)
        - action higher than storage/cash limit
        '''
        reward = 0
        delay_modifier = (self.current_step / MAX_STEPS)

        # empty out spoiled products
        sum_spoiled_product = self.current_inventory.loc[self.current_inventory['time_in_storage'] >
                                                         self.product_shelf_life, 'amount'].sum()
        self.current_inventory = self.current_inventory.loc[self.current_inventory['time_in_storage'] <=
                                                            self.product_shelf_life]

        # penalise for wastage
        reward -= sum_spoiled_product / 1000
        logger.debug(f'The spoiled product reward: {-sum_spoiled_product / 1000}')

        # penalise inventory below threshold
        if self.current_inventory['amount'].sum() < self.min_inventory_threshold:
            reward -= 1 #* (self.current_inventory - self.min_inventory_threshold)
            logger.debug('Under min inventory: -1')

        # penalise negative inventory
        if self.current_inventory['amount'].sum() < 0:
            reward -= 1
            logger.debug('Under 0 inventory: -1')

        
        # penalise buys too large for storage or cash
        if action > self.cash_buy_limit or action > self.storage_buy_limit:
            reward -= 1
            logger.debug('Outside buy limits: -1')

        # reward buying at the correct price point
        current_price = self.df.loc[self.current_step, "y"]
        next_week_price = self.df.loc[self.current_step + 1, "y"]
        price_profit = next_week_price - current_price

        reward += price_profit * action / self.action_scaler / self.price_diff_scaler

        # punishment for missed price opportunity
        buy_amount_weight = (1 - action / self.action_scaler)
        reward -= price_profit / (1 + action / self.action_scaler) / self.price_diff_scaler * buy_amount_weight

        logger.debug(f'The price profit reward: {price_profit * action / self.action_scaler / self.price_diff_scaler}')
        logger.debug(f'The missed opportunity reward: {-price_profit / (1 + action / self.action_scaler) / self.price_diff_scaler}')

        # generate next observation
        obs = self._next_observation()

        # stop simulation if inventory goes negative
        # done = True if self.current_inventory['amount'].sum() == 0 else False  # TODO: set end signal
        done = False

        if self.save_results:
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

        # add delay modifier to stimulate long-term behaviour
        # reward *= delay_modifier
        # logger.debug((f'The delay modifier {delay_modifier}'))
        logger.debug(f'reward {reward}')
        
        self.counter += 1

        self.current_inventory['time_in_storage'] = self.current_inventory['time_in_storage'] + 1
        logger.debug(f'The inventory: {self.current_inventory["amount"].sum()}')

        return obs, reward, done, {}

    def _take_action(self, action: float):
        """Update state parameters based on the action."""

        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "y"]

        amount = action
        logger.debug(f'action {amount}')

        # determine max amount of buyable product (storage and cash constraints)
        product_bought = min([self.storage_buy_limit, self.cash_buy_limit, amount])
        logger.debug(f'product bought {product_bought}')

        
        # calculate average buying price of previous products
        # prev_cost = self.cost_basis * self.current_inventory

        # calculate cost of current acquisition
        additional_cost = product_bought * (current_price + self.ordering_cost)
        logger.debug(f'additional cost {additional_cost}')


        # update balance
        self.balance -= additional_cost

        # calculate average buying price of all products
        # self.cost_basis = (prev_cost + additional_cost) / (self.current_inventory + product_bought)

        # update inventory
        if not product_bought == 0:
            self.current_inventory.loc[self.counter] = [0, product_bought]

        self.total_spent_value += additional_cost

        if self.current_inventory['amount'].sum() == 0:
            self.cost_basis = 0

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

        logger.debug(f'balance: {self.balance}')
        logger.debug(f'current price: {current_price}')

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""

        logger.info(f'Step: {self.current_step}')
        logger.info(f'Balance: {self.balance}')
        logger.info(f'product held: {self.current_inventory["amount"].sum()}')
        logger.info(f'Avg cost for held product: {self.cost_basis} (Total spent value: {self.total_spent_value})')

    def plot_buys(self):
        """Plot the behaviour of the buyer agent."""

        # cut off all trailing zeros
        buys_per_step = self.buy_tracker[:self.counter]

        f, ax = plt.subplots()
        plt.title('Buys per time step')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        plt.plot(buys_per_step[:, 0], df.loc[buys_per_step[:, 0], 'y'], linewidth=0.1)
        points = ax.scatter(buys_per_step[:, 0], df.loc[buys_per_step[:, 0], 'y'],
                            marker='o', c=buys_per_step[:, 1], cmap='copper')
        f.colorbar(points)

    def plot_rewards(self):
        """Plot the behaviour of the buyer agent."""

        # cut off all trailing zeros
        reward_per_step = self.reward_tracker[:self.counter]
        # print('max reward: ', max([0.001, max(reward_per_step[:, 0])]))
        # print('min reward: ', min([-0.001, min(reward_per_step[:, 0])]))
        # divnorm_reward = colors.TwoSlopeNorm(vmin=min([-0.001, min(reward_per_step[:, 0])]), vcenter=0,
        #                               vmax=max([0.001, max(reward_per_step[:, 0])]))
        divnorm_reward = colors.TwoSlopeNorm(vmin=-10, vcenter=0,
                                      vmax=10)
        f_reward, ax_reward = plt.subplots()
        plt.title('reward per time step')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        plt.plot(reward_per_step[:, 0], df.loc[reward_per_step[:, 0], 'y'], linewidth=0.1)
        points = ax_reward.scatter(reward_per_step[:, 0], df.loc[reward_per_step[:, 0], 'y'],
                            marker='o', c=reward_per_step[:, 1], cmap='RdYlGn', norm=divnorm_reward)
        f_reward.colorbar(points)

    def plot_inventory(self):
        """Plot the behaviour of the buyer agent."""

        # cut off all trailing zeros
        inventory_per_step = self.inventory_tracker[:self.counter]
        # print('max inven: ', max([0.001, max(inventory_per_step[:, 0])]))
        # print('min inven: ', min([-0.001, min(inventory_per_step[:, 0])]))
        # divnorm_inventory = colors.TwoSlopeNorm(vmin=min([-0.001, min(inventory_per_step[:, 0])]), vcenter=0,
        #                               vmax=max([0.001, max(inventory_per_step[:, 0])]))
        # divnorm_inventory = colors.TwoSlopeNorm(vmin=0,
        #                               vmax=self.storage_capacity)

        logger.debug(f'The inventory: \n {self.inventory_tracker}')

        f_inventory, ax_inventory = plt.subplots()
        plt.title('Inventory per time step')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        plt.plot(inventory_per_step[:, 0], df.loc[inventory_per_step[:, 0], 'y'], linewidth=0.1)
        points = ax_inventory.scatter(inventory_per_step[:, 0], df.loc[inventory_per_step[:, 0], 'y'],
                            marker='o', c=inventory_per_step[:, 1], cmap='RdYlGn')
        f_inventory.colorbar(points)

    def set_saving(self, saving_mode=False):
        """Turn saving on or off."""
        self.save_results = saving_mode


if __name__ == '__main__':
    args = utils.parse_config()
    utils.create_logger_and_set_level(args.verbose)

    df = pd.read_csv('./data/US_SMP_food_TA.csv', index_col=0).iloc[69:].reset_index(drop=True)
    df = df.sort_values('ds')

    # define buyer properties
    properties = {
        'product_shelf_life': 13,
        'ordering_cost': 0.1,
        'storage_capacity': 40000,
        'min_inventory_threshold': 4000,
        'consumption_rate': 3000,
        'storage_cost': 0.2,
        'cash_inflow': 8000000
    }

    ts_feature_names = \
        ["y", "y_24_quo", "y_26_quo", "y_37_quo", "y_94_quo", "y_20_quo", "y_6_quo", "y_227_pro", "y_785_end",
         "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band", "momentum1", "kalman"]

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: BuyerEnvironment(args, df, properties, ts_feature_names)])

    # specify the model used for learning a policy
    # model = PPO("MultiInputLstmPolicy", env, verbose=200)
    model = PPO('MultiInputPolicy', env, verbose=20, learning_rate=0.01)

    # train model
    start_training_time = time.time()
    model.learn(total_timesteps=40000)
    end_training_time = time.time()

    logger.info(f'Total time to train: {end_training_time - start_training_time:.2f} seconds.')

    # carry out simulation
    env.env_method('set_saving', saving_mode=True)
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if i % 200 == 0:
            env.render()

    env.env_method('plot_buys')
    env.env_method('plot_rewards')
    env.env_method('plot_inventory')

    plt.show()


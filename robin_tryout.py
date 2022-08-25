"""File containing the first draft of the Buyer Agent.
Created for the R&D C7 - RL trading agent project.
"""

import gym
import json
import random

import numpy as np
import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym import spaces
from gym.wrappers import FlattenObservation
from sb3_contrib import RecurrentPPO

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_PRODUCT = 2147483647
MAX_PRODUCT_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 25000000


class BuyerEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, properties: dict, ts_feature_names: list):
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
        self.action_space = spaces.Box(low=0, high=20000, shape=(1,))

        self.lookback_period = 20
        # self.observation_space = spaces.Box(low=0, high=1, shape=(21, self.lookback_period + 1), dtype=np.float16)
        self.observation_space = spaces.Dict({
            'time_series': spaces.Box(low=0, high=np.inf, shape=(20, self.lookback_period)),
            'env_props': spaces.Box(low=0, high=1, shape=(4,))
        })

        self.product_shelf_life = properties['product_shelf_life']
        self.ordering_cost = properties['ordering_cost']
        self.storage_capacity = properties['storage_capacity']
        self.min_inventory_threshold = properties['min_inventory_threshold']
        self.consumption_rate = properties['consumption_rate']
        self.storage_cost = properties['storage_cost']
        self.cash_inflow = properties['cash_inflow']
        self.buy_tracker = np.zeros((100000, 2))
        self.counter = 0

    def reset(self) -> dict:
        """Reset the state of the environment to an initial state.

        Returns:
            Initial observation to start with
        """
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.current_inventory = self.min_inventory_threshold * 2
        self.cost_basis = 0
        self.total_spent_value = 0
        self.reward = None

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'y'].values) - (self.lookback_period + 1))
        return self._next_observation()

    def _next_observation(self) -> dict:
        """Create observation dictionary with all features.

        Returns:
            Next inputs for model
        """

        # Get the data points for the last 'look_back' weeks and scale to between 0-1
        frame = {
            'time_series': [
                self.df.iloc[self.current_step: self.current_step + self.lookback_period, :][feat_name].values
                for feat_name in self.ts_feature_names
            ],
            'env_props': [
                self.balance / MAX_ACCOUNT_BALANCE,
                self.current_inventory / MAX_NUM_PRODUCT,
                self.cost_basis / MAX_PRODUCT_PRICE,
                self.total_spent_value / (MAX_NUM_PRODUCT * MAX_PRODUCT_PRICE),
        ]}

        # Append additional data and scale each value to between 0-1

        return frame

    def step(self, action: float) -> [dict, float, bool, dict]:
        """Take the next action and update observations and rewards."""
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'y'].values) - self.lookback_period:
            self.current_step = 0
        delay_modifier = (self.current_step / MAX_STEPS)

        if not self.reward:
            reward = 10000

        # add a penalty
        if self.current_inventory < self.min_inventory_threshold:
            reward += 1000000 * (self.current_inventory - self.min_inventory_threshold)

        if self.current_inventory < 0:
            reward -= 100000

        obs = self._next_observation()
        done = False  # TODO: change

        # print(self.buy_tracker, self.buy_tracker[self.counter])
        # print([self.current_step, action])
        self.buy_tracker[self.counter] = np.array([self.current_step, action[0]])
        self.counter += 1

        return obs, reward, done, {}

    def _take_action(self, action: float):
        """Update state parameters based on the action."""

        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "y"]

        amount = action
        # print(amount)

        # determine max amount of buyable product (storage and cash constraints)
        cash_buy_limit = int(self.balance / current_price)
        storage_buy_limit = self.storage_capacity - self.current_inventory
        product_bought = min([storage_buy_limit, cash_buy_limit, amount])

        # calculate average buying price of previous products
        prev_cost = self.cost_basis * self.current_inventory

        # calculate cost of current acquisition
        additional_cost = product_bought * (current_price + self.ordering_cost)

        # update balance
        self.balance -= additional_cost

        # calculate average buying price of all products
        self.cost_basis = (prev_cost + additional_cost) / (self.current_inventory + product_bought)

        # update inventory
        self.current_inventory += product_bought

        self.total_spent_value += additional_cost

        if self.current_inventory == 0:
            self.cost_basis = 0

        if self.current_inventory < 0:
            self.current_inventory = 0

        # update inventory with spoiled and used product
        self.current_inventory -= self.consumption_rate

        # update balance with operational costs
        self.balance += self.cash_inflow - self.storage_cost * self.current_inventory

    def render(self, mode='human', close=False):
        """Render the environment to the screen."""

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'product held: {self.current_inventory}')
        print(f'Avg cost for held product: {self.cost_basis} (Total spent value: {self.total_spent_value})')

    def plot(self):
        """Plot the behaviour of the buyer agent."""

        # cut off all trailing zeros
        buys_per_step = self.buy_tracker[:self.counter]

        f, ax = plt.subplots()
        plt.title('Buys per time step')
        plt.xlabel('Time step')
        plt.ylabel('Bought product')
        points = ax.scatter(buys_per_step[:, 0], df.loc[buys_per_step[:, 0], 'y'], marker='o', c=buys_per_step[:,1], cmap="plasma")
        f.colorbar(points)
        plt.show()


if __name__ == '__main__':

    df = pd.read_csv('./data/US_SMP_food_TA.csv', index_col=0).iloc[69:].reset_index(drop=True)
    df = df.sort_values('ds')

    # define buyer properties
    properties = {
        'product_shelf_life': 8,
        'ordering_cost': 0.1,
        'storage_capacity': 40000,
        'min_inventory_threshold': 4000,
        'consumption_rate': 3000,
        'storage_cost': 0.1,
        'cash_inflow': 7500000
    }

    ts_feature_names = \
        ["y", "y_24_quo", "y_26_quo", "y_37_quo", "y_94_quo", "y_20_quo", "y_6_quo", "y_227_pro", "y_785_end",
         "ma4", "var4", "momentum0", "rsi", "MACD", "upper_band", "ema", "diff4", "lower_band", "momentum1", "kalman"]

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: FlattenObservation(BuyerEnvironment(df, properties, ts_feature_names))])

    # specify the model used for learning a policy
    # model = RecurrentPPO("MlpLstmPolicy", env, verbose=20)
    model = PPO('MlpPolicy', env, verbose=20)

    # train model
    model.learn(total_timesteps=10000)


    # carry out simulation
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if i % 200 == 0:
            env.render()

    env.env_method('plot')
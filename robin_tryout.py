import gym
import json
import random


import numpy as np
import datetime as dt
import pandas as pd

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym import spaces

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

class BuyerEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df, properties):
        super(BuyerEnvironment, self).__init__()
        self.df = df
        self.properties = properties
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, or refrain, etc.
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 1000]))
        # Prices contains the OHCL values for the last five prices

        self.lookback_period = 5
        self.observation_space = spaces.Box(low=0, high=1, shape=(21, self.lookback_period + 1), dtype=np.float16)

        self.product_shelf_life = properties['product_shelf_life']
        self.ordering_cost = properties['ordering_cost']
        self.storage_capacity = properties['storage_capacity']
        self.min_inventory_threshold = properties['min_inventory_threshold']
        self.consumption_rate = properties['consumption_rate']
        self.storage_cost = properties['storage_cost']
        self.cash_inflow = properties['cash_inflow']

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.current_inventory = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'y'].values) - (self.lookback_period + 1))
        return self._next_observation()


    def _next_observation(self):
        # Get the data points for the last 5 weeks and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_24_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_26_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_37_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_94_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_20_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_6_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_227_pro'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'y_785_end'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'ma4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'var4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'momentum0'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'rsi'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'MACD'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'upper_band'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'ema'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'diff4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'lower_band'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'momentum1'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           self.lookback_period, 'kalman'].values / MAX_SHARE_PRICE,
        ])


        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.current_inventory / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)


        return obs

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'y'].values) - 6:
            self.current_step = 0
        delay_modifier = (self.current_step / MAX_STEPS)

        reward = 10000

        # add a penalty
        if self.current_inventory < self.min_inventory_threshold:
            reward += 1000 * (self.current_inventory - self.min_inventory_threshold)

        done = self.net_worth <= 0
        obs = self._next_observation()
        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "y"]

        action_type = action[0]
        amount = action[1]

        # Buy amount of product
        if action_type < 1:

            # determine max amount of buyable product (storage and cash constraints)
            total_possible_cash = int(self.balance / current_price)
            total_possible_storage = self.storage_capacity - self.current_inventory

            total_possible = min([total_possible_storage, total_possible_cash])

            # buy the amount, if too much, buy max amount
            if amount < total_possible:
                shares_bought = amount
            else:
                shares_bought = total_possible

            # calculate average buying price of previous products
            prev_cost = self.cost_basis * self.current_inventory

            # calculate cost of current acquisition
            additional_cost = shares_bought * current_price + shares_bought * self.ordering_cost

            # update balance
            self.balance -= additional_cost

            # calculate average buying price of all products
            self.cost_basis = (
                prev_cost + additional_cost) / (self.current_inventory + shares_bought)

            # update inventory
            self.current_inventory += shares_bought

        # don't buy
        elif action_type < 2:
            pass

        self.net_worth = self.balance + self.current_inventory * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.current_inventory == 0:
            self.cost_basis = 0

        # update inventory with spoiled and used product
        self.current_inventory -= self.consumption_rate

        # update balance with operational costs
        self.balance += self.cash_inflow - self.storage_cost * self.current_inventory

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.current_inventory} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')


if __name__ == '__main__':


    df = pd.read_csv('./data/US_SMP_food_TA.csv').iloc[69:].reset_index()
    df = df.sort_values('ds')

    # define buyer properties
    properties = {
        'product_shelf_life': 8,
        'ordering_cost': 0.1,
        'storage_capacity': 200000,
        'min_inventory_threshold': 100,
        'consumption_rate': 200,
        'storage_cost': 0.1,
        'cash_inflow': 1000000
    }

    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: BuyerEnvironment(df, properties)])
    model = PPO("MlpPolicy", env, verbose=20)

    # train model
    model.learn(total_timesteps=20000)

    # carry out simulation
    obs = env.reset()
    for i in range(4000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if i % 200 == 0:
            env.render()
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

class StockTradingEnvironment(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, df):
        super(StockTradingEnvironment, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(21, 6), dtype=np.float16)

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, len(self.df.loc[:, 'y'].values) - 6)
        return self._next_observation()


    def _next_observation(self):
        # Get the data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_24_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_26_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_37_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_94_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_20_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_6_quo'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_227_pro'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'y_785_end'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'ma4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'var4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'momentum0'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'rsi'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'MACD'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'upper_band'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'ema'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'diff4'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'lower_band'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'momentum1'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                                           5, 'kalman'].values / MAX_SHARE_PRICE,
        ])


        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
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

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0
        obs = self._next_observation()
        return obs, reward, done, {}

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = self.df.loc[self.current_step, "y"]

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')


if __name__ == '__main__':


    df = pd.read_csv('./data/US_SMP_food_TA.csv').iloc[69:].reset_index()
    df = df.sort_values('ds')
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnvironment(df)])
    model = PPO("MlpPolicy", env, verbose=20)
    model.learn(total_timesteps=100000)
    obs = env.reset()
    for i in range(4000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if i % 200 == 0:
            env.render()
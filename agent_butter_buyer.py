"""File containing the first draft of the Buyer Agent.
Created for the R&D C7 - RL trading agent project.
"""
from typing import Union
import gym
import random
import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors

from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from gym import spaces
from gym.wrappers import FlattenObservation
from sb3_contrib import RecurrentPPO

import utils

logger = logging.getLogger('logger')

MAX_STEPS = 20000
INITIAL_ACCOUNT_BALANCE = 10000000

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
        self.counter = 0
        self.reward = []
        self.save_results = save_results

        # parameters
        self.upper_buy_limit = properties['upper_buy_limit']
        self.price_diff_scaler = 10 # price difference on range [0, 50]

    def data_scaler(self, df):     
        scale_columns = np.setdiff1d(df.columns.values, ['ds'])
        df_to_scale = df[scale_columns]
        df_scaled = (df_to_scale-df_to_scale.min())/(df_to_scale.max()-df_to_scale.min())*2 - 1
        df_scaled.insert(0, 'ds', df.ds)
        return df_scaled
        
        
    def reset_values(self):
        """Reset simulation values to starting values."""
        self.balance = INITIAL_ACCOUNT_BALANCE

        self.total_spent_value = 0
        self.counter = 0
        self.current_inventory = pd.DataFrame(columns=['time_in_storage', 'amount'], dtype=float)
        self.current_inventory.loc[self.counter] = [0, self.min_inventory_threshold*6]

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
        self.current_step = random.randint(0, len(self.df.loc[:, 'y'].values) - (self.lookback_period + 1))
        self.start_step = self.current_step

        return self._next_observation()
    
    def baseline_reset(self) -> dict:
        """Prepare class for baseline simulation."""
        self.current_step = self.start_step
        self.reset_values()


    def _next_observation(self) -> dict:
        """Create observation dictionary with all features.

        Returns:
            Next inputs for model
        """
        current_price = self.df_y.loc[self.current_step, "y"]
        self.cash_buy_limit = int(self.balance / current_price)
        self.storage_buy_limit = self.storage_capacity - self.current_inventory['amount'].sum()

        # Get the data points for the last 'look_back' weeks and scale to between 0-1
        frame = {
            'time_series': [
                self.df.iloc[self.current_step: self.current_step + self.lookback_period, :][feat_name].values
                for feat_name in self.ts_feature_names
            ],
            'env_props': [
                self.balance/100000000000,
                self.current_inventory['amount'].sum()/self.storage_capacity,
                min([self.cash_buy_limit/self.storage_capacity, 1]),
                self.storage_buy_limit/self.storage_capacity,                
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
        """Take the next action and update state and rewards."""
        logger.debug("---------------")
        logger.debug(f'Step: {self.current_step}')

        # multiply action [-1,1] by upper_buy_limit factor (works better with NN model)
        action_buy_amount = action[0]*self.upper_buy_limit
        self._take_action(action_buy_amount)

        # make sure the current step does not move past the end of the ts
        self.current_step += 1
        if self.current_step > len(self.df.loc[:, 'y'].values) - self.lookback_period:
            self.current_step = 0
        '''
        reward
        - buy at the right price
        punish
        - missed opportunity
        - spoilage
        - inventory below threshold
        - inventory below 0 
        - action higher than storage/cash limit
        '''
        inv_under_min_reward = 0
        inv_under_zero_reward = 0
        action_over_limit_reward = 0

        # reward buying at the correct price point
        current_price = self.df_y.loc[self.current_step, "y"]
        next_week_price = self.df_y.loc[self.current_step + 1, "y"]
        price_profit = next_week_price - current_price
        buy_priceprofit_reward = price_profit * action / self.upper_buy_limit / self.price_diff_scaler #NOTE maybe product_bought better than action, test

        # punishment for missed price opportunity
        buy_amount_weight = (0.3 - action / self.upper_buy_limit)
        missed_opportunity_reward = price_profit / (1 + action / self.upper_buy_limit) / self.price_diff_scaler * buy_amount_weight

        logger.debug(f'current price: {current_price}')
        logger.debug(f'next week price: {next_week_price}')
        logger.debug(f'The price profit reward: {buy_priceprofit_reward}')
        logger.debug(f'The missed opportunity reward: {-missed_opportunity_reward}')
        
        # punishment for spoilage + update inventory
        sum_spoiled_product = self.current_inventory.loc[self.current_inventory['time_in_storage'] >
                                                         self.product_shelf_life, 'amount'].sum()
        spoil_reward = sum_spoiled_product / 1000
        logger.debug(f'The spoiled product reward: {-spoil_reward}')
        
        self.current_inventory = self.current_inventory.loc[self.current_inventory['time_in_storage'] <=
                                                            self.product_shelf_life]

        # punishment for inventory below threshold
        if self.current_inventory['amount'].sum() < self.min_inventory_threshold:
            inv_under_min_reward = 1 #* (self.current_inventory - self.min_inventory_threshold)
            logger.debug('Under min inventory: -1')

        # punishment for negative inventory
        if self.current_inventory['amount'].sum() < 0:
            inv_under_zero_reward = 1
            logger.debug('Under 0 inventory: -1')

        # punishment for buy action too large for storage or cash limits
        if action_buy_amount > self.cash_buy_limit or action > self.storage_buy_limit:
            action_over_limit_reward = 1
            logger.debug('Action over buy/storage limits: -1')

        # add delay modifier to stimulate long-term behaviour
        # reward *= delay_modifier
        
        # calculate total reward
        reward = buy_priceprofit_reward - missed_opportunity_reward - spoil_reward - inv_under_min_reward - inv_under_zero_reward - action_over_limit_reward
        logger.debug(f'Reward {reward}')
        logger.debug(f'New inventory: {self.current_inventory["amount"].sum()}')
        
        if self.save_results:
            self.update_measure_tracker(action_buy_amount, reward)
        
        # generate next observation
        obs = self._next_observation()

        # stop simulation if inventory goes negative
        # done = True if self.current_inventory['amount'].sum() == 0 else False  # TODO: set end signal
        done = False
        
        self.counter += 1
        self.current_inventory['time_in_storage'] = self.current_inventory['time_in_storage'] + 1

        return obs, reward, done, {}

    def _take_action(self, action_buy_amount: float):
        """Update state parameters based on the action."""

        # Set the current price to a random price within the time step
        current_price = self.df_y.loc[self.current_step, "y"]

        logger.debug(f'Action {action_buy_amount}')

        # determine max amount of buyable product (storage and cash constraints)
        self.product_bought = min([self.storage_buy_limit, self.cash_buy_limit, action_buy_amount])
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
        total_worth = self.balance + current_inventory * self.df_y.loc[self.current_step, "y"]
        return total_worth, self.balance, current_inventory, self.total_spent_value
        
    def plot_measure(self, measure='buys'):
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
        
        f, ax = plt.subplots()
        plt.title(f'{measure} per time step')
        plt.xlabel('Time step')
        plt.ylabel('Price of product')
        plt.plot(measure_per_step[:, 0], self.df_y.loc[measure_per_step[:, 0], 'y'], linewidth=0.1)
        points = ax.scatter(measure_per_step[:, 0], self.df_y.loc[measure_per_step[:, 0], 'y'],
                            marker='o', c=measure_per_step[:, 1], cmap=cmap, norm=norm)
        f.colorbar(points)


    def set_saving(self, saving_mode=False):
        """Turn saving on or off."""
        self.save_results = saving_mode
    
def run_simulation(env, plot=False):
    """Run simulation of trained model."""
    if plot:
        env.env_method('set_saving', saving_mode=True)
    
    logger.setLevel(logging.DEBUG)
    obs = env.reset()
    for i in range(args.simsteps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if i % 200 == 0:
            env.render()
            
    # show results
    logger.info("Model final results:")
    env.render()
    
    if plot:
        utils.plot_results(env=env)
    return env

def run_baseline_simulation(env, action, steps=1000):
    """Run baseline simulation."""
    logger.setLevel(logging.INFO)

    obs = env.env_method('baseline_reset')
    for i in range(steps):
        _ = env.step([[action]])

    logger.info("Baseline final results")
    env.render()
    return env


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
        'min_inventory_threshold': 4000,
        'consumption_rate': 3000,
        'storage_cost': 0.2,
        'cash_inflow': 8000000,
        'upper_buy_limit': 10000
    }

    # setup vectorized env and model
    env = DummyVecEnv([lambda: BuyerEnvironment(args, df, properties, ts_feature_names)])
    model = PPO('MultiInputPolicy', env, verbose=20, learning_rate=0.01)

    # train model
    utils.run_and_track_runtime(model.learn, total_timesteps=args.trainsteps)

    # carry out simulation
    env = run_simulation(env, plot=args.plot)
    total_worth, \
        balance, \
        current_inventory, \
        total_spent_value = env.env_method('return_results')[0]
    
    # run baseline: buying consumption rate
    env = run_baseline_simulation(env=env, action=properties['consumption_rate']/properties['upper_buy_limit'], steps=args.simsteps)
    b_total_worth, \
        b_balance, \
        b_current_inventory, \
        b_total_spent_value = env.env_method('return_results')[0]
    
    print(f'Total worth (incl inventory) agent butter: {total_worth:.2f}')
    print(f'Total worth (incl inventory) baseline: {b_total_worth:.2f}')

    print(f'Total worth improvement over baseline: {(total_worth-b_total_worth)/b_total_worth*100:.4f}%')
    
    


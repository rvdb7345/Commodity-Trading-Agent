![DALL·E 2022-11-02 11 43 58 - A futuristic poster of an AI trading agent buying dairy commodities using an online platform](https://user-images.githubusercontent.com/33393959/199487657-dda11f14-6bd7-4ad7-bc54-c54e38fe0bfd.png)

# RL Commodity Trading Agent

This repo provides a modular environment for training an agent to handle the procurement of agricultural commodities through the use of Reinforcement Learning. Created by Vesper in collobaration with Slimmer AI.

Check out the website for more information about our platform: https://www.vespertool.com/

## Current data
Current data added to the repo is:
- `US_SMP_food_TA.csv` --> example data of the US Skimmed Milk Powder price including a set of Technical Indicators.


## Installation

_Before env installation:_ Make sure that openmpi is installed. If not yet installed, install by:

For Mac:
```commandline
brew install open-mpi
```
As outlined in the documentation of mpi4py (for which open-mpi is needed), check if the mpicc compiler wrapper is on 
your search path by running `echo $PATH`. If not, add it by running `export PATH=$PATH:~/path/to/mpicc` for mac users 
the path is something like: "opt/homebrew/bin/mpicc"

For ubuntu:
```
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Then install pip environment:
```commandline
pipenv install
```

Now all requirements should be installed. Installation can also go through installation of the requirements.txt 
(if kept up-to-date).

## Training and Using the Agent
All code for running our agent SMP buyer can be found in `agent_SMP_buyer.py`. Running this file will now use the 
PPO actor-critic model from the stable baselines package and compare results with a simple baseline 
(baseline strategy: buy what is needed every week).

### Flags
#### --verbose [str], -v [str]
This flag can be used to set the logger level, by default this is set to `info` level which logs basic information on 
training and simulating step. The `debug` level shows multiple measures/rewards per step.

#### --plot, -p
Enables plotting of rewards, inventory, actions of the simulation after training.

#### --trainsteps [int], -t [int]
Determines the number of training steps, default is 40000 which takes ±1 minute. Running longer (>10min) is 
recommended for better results.

#### A command line example
```commandline
python agent_SMP_buyer.py -v "debug" -t 20000 -p
```



### Evaluating the algorithm
To evaluate the robustness of the RL algorithm, `experiment_SMP_buyer.py` can be used. It repeats the training and 
simulation of the model and then evaluates the significance of the difference between train and test scores and their 
baselines through Wilcoxon sign tests.

#### command line example
```commandline
python experiment_SMP_buyer.py -t 40000 -r 30
```

#### --reps [int], -r [int]
Determines the number of experiment repetitions to estimate the robustness of the RL algorithm training.

### Evaluating the influence of a parameter
To evaluate the influence of a specific parameter, this repo provides functionality to do sensitivity analysis. This is done in 
sensitivity_analysis.py. Here, a value span can be given and the base parameters can be changed. 
Subsequently, you can set the reps and training steps using the appropriate flags and run the analysis using:

```commandline
python sensitivity_analysis.py -t 40000 -r 30
```

After creating the results for every parameter, it can be visualised using the following command:

```commandline
python visualise_SA.py
```


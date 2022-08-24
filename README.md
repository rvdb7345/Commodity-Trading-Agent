# C7-RL-buyer

A repo for the buyer persona in the Reinforcement Learning project of C7. 

##### Current data
Current data added to the repo is:
- AAPL.csv --> stock market data of the AAPL stock for experimentation.
- US_SMP_food_TA.csv --> example data of the US Skimmed Milk Powder price including prices of related products/countries and Technical Indicators.


##### Installation

_Before env installation:_ Make sure that openmpi is installed. If not yet installed, install by:

For Mac:
```commandline
brew install cmake openmpi
```

For ubuntu:
```commandline
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

Then install pip environment:
```commandline
pipenv install
```

Now all requirements should be installed. Installation can also go through installation of the requirements.txt 
(if kept up-to-date).
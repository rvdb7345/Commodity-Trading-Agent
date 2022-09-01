import argparse
import logging
import time
from types import SimpleNamespace
from matplotlib import pyplot as plt, colors


logging.getLogger('matplotlib').setLevel(logging.ERROR)
logger = logging.getLogger('logger')


def parse_config():
    """Config parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', nargs='?', const='info', type=str, default='info',
                        help='Print extra info, levels: debug, info, warning, critical, error')
    parser.add_argument('--plot', '-p', action='store_true', default=False, 
                        help='plot results')
    parser.add_argument('--trainsteps', '-t', default=40000, type=int, 
                        help='training steps')
    parser.add_argument('--simsteps', '-s', default=1000, type=int,
                        help='simulation steps')
    args = vars(parser.parse_known_args()[0])
    args = SimpleNamespace(**args)
    return args

def create_logger_and_set_level(logging_level):
    """Set the requested logging level."""
    if logging_level.lower() == 'info':
        verbosity = logging.INFO
    elif logging_level.lower() == 'warning':
        verbosity = logging.WARNING
    elif logging_level.lower() == 'debug':
        verbosity = logging.DEBUG
    elif logging_level.lower() == 'critical':
        verbosity = logging.CRITICAL
    elif logging_level.lower() == 'error':
        verbosity = logging.ERROR
    else:
        logging.warning(f'Verbosity level {logging_level} not recognised, defaulted to info.')
        verbosity = logging.INFO

    logging.getLogger('matplotlib').setLevel(logging.ERROR)

    logging.basicConfig(
        level=verbosity,
        format='(%(asctime)s.%(msecs)03d): %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    
def run_and_track_runtime(func, **args):
    start_time = time.time()
    func(**args)
    end_time = time.time()
    logger.info(f'Total time to train: {end_time - start_time:.2f} seconds.')
    
def plot_results(env):
    env.env_method('plot_measure', measure='buys')
    env.env_method('plot_measure', measure='reward')
    env.env_method('plot_measure', measure='inventory')
    plt.show()

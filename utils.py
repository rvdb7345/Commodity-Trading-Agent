import argparse
import logging
from types import SimpleNamespace

def parse_config():
    """Config parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', nargs='?', const='info', type=str, default='info',
                        help='Print extra info, levels: debug, info, warning, critical, error')
    parser.add_argument('--plot', '-p', action='store_true', default=False, 
                        help='plot results')
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

    logging.basicConfig(
        level=verbosity,
        format='(%(asctime)s.%(msecs)03d): %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
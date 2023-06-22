import logging
from pathlib import Path

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

if not (Path(__file__).parent / 'settings.py').exists():
    raise ValueError('Cannot find settings.py file')

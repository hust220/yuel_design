import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.disc2.model import DiscModel
from src.disc2.dataset import DiscDataset
import src.disc2.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=DiscModel, dataset=DiscDataset)

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.coords2.model import CoordsModel
from src.coords2.dataset import CoordsDataset
import src.coords2.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=CoordsModel, dataset=CoordsDataset)

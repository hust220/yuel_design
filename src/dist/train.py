import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.dist.model import DistModel
from src.dist.dataset import DistDataset
import src.dist.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=DistModel, dataset=DistDataset)

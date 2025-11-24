import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.cont2.model import ContModel
from src.cont2.dataset import ContDataset
import src.cont2.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=ContModel, dataset=ContDataset)

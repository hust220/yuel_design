import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sidechain.model import SidechainModel
from src.sidechain.dataset import SidechainDataset
import src.sidechain.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=SidechainModel, dataset=SidechainDataset)

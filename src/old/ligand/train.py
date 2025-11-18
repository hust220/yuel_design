import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.ligand.model import LigandModel
from src.ligand.dataset import LigandDataset
import src.ligand.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=LigandModel, dataset=LigandDataset)

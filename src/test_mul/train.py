import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.test_mul.model import E2EModel
from src.test_mul.dataset import E2EDataset
import src.test_mul.config as config
from src.utils import run_training

if __name__ == '__main__':
    run_training(config=config, model=E2EModel, dataset=E2EDataset)

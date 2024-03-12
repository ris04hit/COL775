import os
import sys

sys.path.append(os.path.abspath('../'))
from src.utils import create_dir

create_dir("../temp")
create_dir("../data")
create_dir("../data/processed")
create_dir("../data/processed/train")
create_dir("../data/processed/test")
create_dir("../data/processed/val")
create_dir("../data/raw")
create_dir("../data/raw/train")
create_dir("../data/raw/test")
create_dir("../data/raw/val")
create_dir("../results")
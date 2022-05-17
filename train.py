import sys, os
import numpy as np
import tensorflow as tf
import logging
from models.JMOD2 import JMOD2
from lib.trainer import Trainer
from config import get_config
from lib.utils import prepare_dirs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
config = None

def main():
	# Log dirs
	prepare_dirs(config)

    # RNG initialization
	rng = np.random.RandomState(config.random_seed)
	tf.random.set_seed(config.random_seed)

	# Instantiate the JMOD2 model
	model = JMOD2(config)

	# Prepare trainer
	trainer = Trainer(config, model, rng)

	# train
	if config.resume_training:
		trainer.resume_training()
	else:
		trainer.train()

if __name__ == "__main__":
    config, unparsed = get_config()
    main()

import time

import config
from train import main
from logger import logger

if __name__ == "__main__":
    """
    /!\ if rootdirectory isn't cycleGAN/, then it won't work /!\ 
    """
    START_TIME = time.time()

    config.HORSES_CLASS = "glasses"
    config.ZEBRAS_CLASS = "no_glasses"
    config.SKIP_CONNECTION = 0
    config.SIZE = 32
    config.LAMBDA_IDENTITY = 0
    config.ONE_SIDED_LABEL_SMOOTHING = 0
    config.REPETITION_NUMBER = 1
    config.NUM_EPOCHS = 100
    config.SAUVEGARDE_TOUS_LES_CB = 1
    config.HOW_MANY_VAL_SAVED = 5

    config.def_transforms()

    main(START_TIME)

    logger(
        f"Total time {time.time() - start_time}",
        True,
        str(time.time() - start_time),
        "total_time"
    )

import time
import argparse

import config
from train import main
from logger import logger

if __name__ == "__main__":
    """
    /!\ if rootdirectory isn't cycleGAN/, then it won't work /!\ 
    """
    START_TIME = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument("horses_class",
                        help='Directory name (in data/train and in data/val) where are the images of class 1.')
    parser.add_argument("zebras_class",
                        help='Directory name (in data/train and in data/val) where are the images of class 2.')
    parser.add_argument("skip_connection",
                        help='Can take values 0 (False), 1 (only the first layer feeds the last layer) or 2 '
                             '(every intermediaite layers during the downsampling process feed into the corresponding '
                             'layers when upsampling.')
    parser.add_argument("size",
                        help='What size images should be resized to (size x size).')
    parser.add_argument("lambda_identity",
                        help='Value of lambda_identity for the loss.')
    parser.add_argument("one_sided_label_smoothing",
                        help='Keeping the labels from real images to be 1 but rather a random float between '
                             '1-one_sided_label_smoothing and 1.')
    parser.add_argument("repetition_number",
                        help='What folder should we write in?')
    parser.add_argument("num_epoch",
                        help='Number of epochs during which the network is trained.')
    parser.add_argument("sauvegarde_tous_les_cb",
                        help='How often do we save the images of the validation set?')
    parser.add_argument("how_many_val_saved",
                        help='How many validaiton set images are saved?')

    args = parser.parse_args()

    config.HORSES_CLASS = args.horses_class
    config.ZEBRAS_CLASS = args.zebras_class
    config.SKIP_CONNECTION = int(args.skip_connection)
    config.SIZE = int(args.size)
    if config.SIZE > 512:
        config.BATCH_SIZE = 1
    elif config.SIZE > 256:  # i.e. between 256 & 512 because of the elif clause
        config.BATCH_SIZE = 3
    config.LAMBDA_IDENTITY = float(args.lambda_identity)
    config.ONE_SIDED_LABEL_SMOOTHING = float(args.one_sided_label_smoothing)
    config.REPETITION_NUMBER = args.repetition_number
    config.NUM_EPOCHS = args.num_epoch
    config.SAUVEGARDE_TOUS_LES_CB = args.sauvegarde_tous_les_cb
    config.HOW_MANY_VAL_SAVED = args.how_many_val_saved

    config.def_transforms()

    main(START_TIME)

    logger(
        f"Total time {time.time() - start_time}",
        True,
        str(time.time() - start_time),
        "total_time"
    )

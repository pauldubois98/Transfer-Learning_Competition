import config
from utils import create_directory


def logger(message: str, stored: bool, to_store: str, output: str) -> None:
    """
    :param message: will be printed
    :param stored: boolean
    :param to_store: will be stored if stored
    :param output: what file will the to_store string will be stored to if stored
    :return: None
    """
    print(message)
    if stored:
        outs_folder_classe = f"outs_{config.REPETITION_NUMBER}/{config.HORSES_CLASS}_{config.ZEBRAS_CLASS}"
        create_directory(outs_folder_classe)

        outs_folder_classe_skipconnections = f"{outs_folder_classe}/skip_{config.SKIP_CONNECTION}"
        create_directory(outs_folder_classe_skipconnections)

        outs_folder_classe_skipconnections_size = f"{outs_folder_classe_skipconnections}/{config.SIZE}"
        create_directory(outs_folder_classe_skipconnections_size)

        outs_folder_classe_skipconnections_size_li = f"{outs_folder_classe_skipconnections_size}/" \
                                                     f"li_{float(config.LAMBDA_IDENTITY)}"
        create_directory(outs_folder_classe_skipconnections_size_li)

        outs_folder_classe_skipconnections_size_li_osls = f"{outs_folder_classe_skipconnections_size_li}/" \
                                                          f"osls_{config.ONE_SIDED_LABEL_SMOOTHING}"
        create_directory(outs_folder_classe_skipconnections_size_li_osls)
        with open(f'{outs_folder_classe_skipconnections_size_li_osls}/{output}.txt', 'a+') as f:
            f.write(to_store)

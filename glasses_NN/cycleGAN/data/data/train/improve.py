import os
import cv2
import argparse
from cv2 import dnn_superres  # Download les modèles https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

def improve(folder: str, new_folder: str, size: int):
    """
    :param folder: origin folder with different image of different sizes (but all squared)
    :param new_folder: folder with images of size = size x size
    :param size: size desired
    :return: nothing
    """
    try:
        os.mkdir(new_folder)
    except FileExistsError:
        pass

    for file in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, file))
        (height, width, channels) = image.shape

        if height == size and width == size:
            print(file, "- ok")
            continue

        if max(height, width) >= size:
            result = image
        elif max(height, width) >= size // 2:
            result = sr2.upsample(image)
        elif max(height, width) >= size // 3:
            result = sr3.upsample(image)
        else:
            result = sr4.upsample(image)

        resized_image = cv2.resize(result, (size, size))

        cv2.imwrite(os.path.join(new_folder, file), resized_image)

        print(file, "- resized")


if __name__ == "__main__":
    print("Script launching (python)")
    sr2 = dnn_superres.DnnSuperResImpl_create()
    sr2.readModel("EDSR_x2.pb")
    sr2.setModel("edsr", 2)
    sr3 = dnn_superres.DnnSuperResImpl_create()
    sr3.readModel("EDSR_x3.pb")
    sr3.setModel("edsr", 3)
    sr4 = dnn_superres.DnnSuperResImpl_create()
    sr4.readModel("EDSR_x4.pb")
    sr4.setModel("edsr", 4)

    parser = argparse.ArgumentParser()

    parser.add_argument("n_array")

    args = parser.parse_args()

    if args.n_array == "0":
        improve("glasses/", "glasses_improved/", 1024)
    elif args.n_array == "1":
        improve("no_glasses/", "no_glasses_improved/", 1024)

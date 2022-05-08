import torch
import random
import sys
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import config
from dataset import HorseZebraDataset
from utils import save_checkpoint, load_checkpoint, create_directory
from discriminator_model import Discriminator
from generator_model import Generator
from logger import logger


def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, val_loader,
             opt_disc, opt_gen, scheduler_disc, scheduler_gen, l1, mse, d_scaler, g_scaler, epoch,
             save_val_images_transformed: bool = False):
    H_reals = 0
    H_fakes = 0

    if not config.ONLY_GENERATE:  # si ONLY_GENERATE on passe directement à la suite (eval)
        # TRAIN
        disc_H.train()
        disc_Z.train()
        gen_Z.train()
        gen_H.train()
        for idx, (zebra, horse) in enumerate(loader):
            # zebra and horses are of size (config.BATCH_SIZE, 3, 256, 256)
            zebra = zebra.to(config.DEVICE)
            horse = horse.to(config.DEVICE)

            # Train Discriminators H and Z
            with torch.cuda.amp.autocast():
                fake_horse = gen_H(zebra)
                D_H_real = disc_H(horse)  # D_H_real c'est 30x30 valeurs qui doivent valoir 1
                D_H_fake = disc_H(fake_horse.detach())  # D_H_fake c'est 30x30 valeurs qui doivent valoir 0
                H_reals += D_H_real.mean().item()  # H_reals c'est la somme des moyennes des valeurs dans la grille 30x30
                H_fakes += D_H_fake.mean().item()  # H_fakes c'est la somme des moyennes des valeurs dans la grille 30x30
                D_H_real_loss = mse(D_H_real,
                                    torch.ones_like(D_H_real) - random.random() * config.ONE_SIDED_LABEL_SMOOTHING)
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                fake_zebra = gen_Z(horse)
                D_Z_real = disc_Z(zebra)
                D_Z_fake = disc_Z(fake_zebra.detach())
                # on ne fait pas Z_real et Z_fake, mais en soit on considère que si ça marche dans un sens,
                # ça marche dans l'autre
                D_Z_real_loss = mse(D_Z_real,
                                    torch.ones_like(D_Z_real) - random.random() * config.ONE_SIDED_LABEL_SMOOTHING)
                D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
                D_Z_loss = D_Z_real_loss + D_Z_fake_loss

                # put it together
                D_loss = (D_H_loss + D_Z_loss)/2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            old_scale_d = d_scaler.get_scale()
            d_scaler.update()

            # Train Generators H and Z
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_H_fake = disc_H(fake_horse)
                D_Z_fake = disc_Z(fake_zebra)
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

                # cycle loss
                cycle_zebra = gen_Z(fake_horse)
                cycle_horse = gen_H(fake_zebra)
                cycle_zebra_loss = l1(zebra, cycle_zebra)
                cycle_horse_loss = l1(horse, cycle_horse)

                if config.LAMBDA_IDENTITY:
                    # identity loss
                    identity_zebra = gen_Z(zebra)
                    identity_horse = gen_H(horse)
                    identity_zebra_loss = l1(zebra, identity_zebra)
                    identity_horse_loss = l1(horse, identity_horse)
                else:
                    identity_horse_loss = 0
                    identity_zebra_loss = 0

                # add all togethor
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    + cycle_zebra_loss * config.LAMBDA_CYCLE
                    + cycle_horse_loss * config.LAMBDA_CYCLE
                    + identity_horse_loss * config.LAMBDA_IDENTITY
                    + identity_zebra_loss * config.LAMBDA_IDENTITY
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            old_scale_g = g_scaler.get_scale()
            g_scaler.update()

            if idx % 100 == 0 or idx == len(loader) - 1:
                print(f"epoch {epoch} / {config.CURRENT_EPOCH + 1 + config.NUM_EPOCHS}, batch {idx} / {len(loader)} "
                      f"H_real={H_reals/(idx+1):.2f} H_fake={H_fakes/(idx+1):.2f}, "
                      f"lr_d = {opt_disc.param_groups[0]['lr']:.8f}, lr_g = {opt_gen.param_groups[0]['lr']:.8f}")
                sys.stdout.flush()

        ### Fin du training de l'epoch, le dire au scheduler qui permet de decay le learning rate
        # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930
        if old_scale_d <= d_scaler.get_scale():
            # d_scaler.update() decreases the scale_factor when optimizer.step() is skipped
            # and if it's skipped we don't want to scheduler_disc.step
            scheduler_disc.step()
        if old_scale_g <= g_scaler.get_scale():
            scheduler_gen.step()

    # EVAL
    if save_val_images_transformed:
        with torch.no_grad():
            disc_H.eval()
            disc_Z.eval()
            gen_Z.eval()
            gen_H.eval()
            for idx, (zebra, horse) in enumerate(val_loader):
                if idx < max(2048, len(val_loader)):  # c'est sur ces images que FID va être calculé
                    # zebra and horses are of size (config.BATCH_SIZE, 3, config.SIZE, config.SIZE)
                    zebra = zebra.to(config.DEVICE)
                    horse = horse.to(config.DEVICE)

                    fake_horse = gen_H(zebra)
                    fake_zebra = gen_Z(horse)

                    class_directory_path = f"saved_images_{config.REPETITION_NUMBER}/" \
                                           f"{config.HORSES_CLASS}_{config.ZEBRAS_CLASS}"
                    create_directory(class_directory_path)

                    skip_connection_path = f"{class_directory_path}/skip_{config.SKIP_CONNECTION}"
                    create_directory(skip_connection_path)

                    size_path = f"{skip_connection_path}/{config.SIZE}"
                    create_directory(size_path)

                    l_identity_path = f"{size_path}/l_identity_{float(config.LAMBDA_IDENTITY)}"
                    create_directory(l_identity_path)

                    osls_path = f"{l_identity_path}/osls_{config.ONE_SIDED_LABEL_SMOOTHING}"
                    create_directory(osls_path)

                    category_path_horses = f"{osls_path}/was_{config.HORSES_CLASS}"
                    create_directory(category_path_horses)
                    category_path_zebras = f"{osls_path}/was_{config.ZEBRAS_CLASS}"
                    create_directory(category_path_zebras)

                    if config.VAL_IMAGES_FORMAT == "both":
                        save_image(torch.cat((horse * 0.5 + 0.5, fake_zebra * 0.5 + 0.5)),
                                   f"{category_path_horses}/{idx}_epoch_{epoch}.png")
                        save_image(torch.cat((zebra * 0.5 + 0.5, fake_horse * 0.5 + 0.5)),
                                   f"{category_path_zebras}/{idx}_epoch_{epoch}.png")
                    elif config.VAL_IMAGES_FORMAT == "only_gen":
                        save_image(fake_zebra * 0.5 + 0.5,
                                   f"{category_path_horses}/{idx}_epoch_{epoch}.png")
                        save_image(fake_horse * 0.5 + 0.5,
                                   f"{category_path_zebras}/{idx}_epoch_{epoch}.png")
                    else:
                        print(f"Fatal Error: config.VAL_IMAGES_FORMAT can only be \"both\" or \"only_gen\" and"
                              f"was set to {config.VAL_IMAGES_FORMAT}")
                        exit()


def main():

    # To save weights or load them
    weights_folder_classe = f"weights_{config.REPETITION_NUMBER}/{config.HORSES_CLASS}_{config.ZEBRAS_CLASS}"
    create_directory(weights_folder_classe)

    weights_folder_classe_skipconnections = f"{weights_folder_classe}/skip_{config.SKIP_CONNECTION}"
    create_directory(weights_folder_classe_skipconnections)

    weights_folder_classe_skipconnections_size = f"{weights_folder_classe_skipconnections}/{config.SIZE}"
    create_directory(weights_folder_classe_skipconnections_size)

    weights_folder_classe_skipconnections_size_li = f"{weights_folder_classe_skipconnections_size}/" \
                                                    f"li_{float(config.LAMBDA_IDENTITY)}"
    create_directory(weights_folder_classe_skipconnections_size_li)

    weights_folder_classe_skipconnections_size_li_osls = f"{weights_folder_classe_skipconnections_size_li}/" \
                                                         f"osls_{config.ONE_SIDED_LABEL_SMOOTHING}"
    create_directory(weights_folder_classe_skipconnections_size_li_osls)

    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        try:
            load_checkpoint(
                f"{weights_folder_classe_skipconnections_size_li_osls}/{config.CHECKPOINT_GEN_H}", gen_H, opt_gen,
                change_current_epoch=True,
            )
            load_checkpoint(
                f"{weights_folder_classe_skipconnections_size_li_osls}/{config.CHECKPOINT_GEN_Z}", gen_Z, opt_gen,
                change_current_epoch=True,
            )
            load_checkpoint(
                f"{weights_folder_classe_skipconnections_size_li_osls}/{config.CHECKPOINT_CRITIC_H}", disc_H, opt_disc,
                change_current_epoch=True,
            )
            load_checkpoint(
                f"{weights_folder_classe_skipconnections_size_li_osls}/{config.CHECKPOINT_CRITIC_Z}", disc_Z, opt_disc,
                change_current_epoch=True,
            )
            print("Loading previous model: success")
            print(f"Last epoch: {config.CURRENT_EPOCH}")
        except Exception as e:
            print(e)
            print("Couldn't load previous model (exception printed above)")

    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=100, gamma=0.1)
    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=100, gamma=0.1)

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR+f"/{config.HORSES_CLASS}", root_zebra=config.TRAIN_DIR+f"/{config.ZEBRAS_CLASS}",
        transform=config.transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR+f"/{config.HORSES_CLASS}", root_zebra=config.VAL_DIR+f"/{config.ZEBRAS_CLASS}",
        transform=config.transforms_val_dataset
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # on fait ça 1 par 1 pour val_loader, pour pouvoir juger de l'avant-après facilement
                       # à noter que s'il y a une classe plus représentée que l'autre, on aura de la répétition
                       # d'un côté (vu qu'on pioche un horse et un zebra et que si on a fini les horses et pas
                       # les zebras, il repioche un horse au pif). Mais on s'en fout, au pire :)
        shuffle=False,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    logger(
        f"Initialisation: {time.time() - start_time} s",
        True,
        str(time.time() - start_time),
        "initialisation_time"
    )

    for epoch in range(config.CURRENT_EPOCH + 1, config.CURRENT_EPOCH + 1 + config.NUM_EPOCHS + 1):
        # On commence à config.CURRENT_EPOCH + 1, et on en fait config.NUM_EPOCHS. Du coup on va jusqu'à
        # config.CURENT_EPOCH + 1 + config.NUM_EPOCHS inclus du coup on va jusqu'à
        # config.CURENT_EPOCH + 1 + config.NUM_EPOCHS + 1 exclus
        start_time_local = time.time()

        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, val_loader,
                 opt_disc, opt_gen, scheduler_disc, scheduler_gen, L1, mse, d_scaler, g_scaler, epoch,
                 save_val_images_transformed=(epoch%100==1))

        logger(
            f"epoch {epoch} time: {time.time() - start_time_local} s",
            True,
            str(time.time() - start_time),
            "epochs_time"
        )

        if config.SAVE_MODEL and (epoch % 10 == 1 or epoch == config.CURRENT_EPOCH + 1 + config.NUM_EPOCHS):
            save_checkpoint(gen_H, opt_gen, epoch,
                            filename=f"{weights_folder_classe_skipconnections_size_li_osls}/"
                                     f"{config.CHECKPOINT_GEN_H}")
            save_checkpoint(gen_Z, opt_gen, epoch,
                            filename=f"{weights_folder_classe_skipconnections_size_li_osls}/"
                                     f"{config.CHECKPOINT_GEN_Z}")
            save_checkpoint(disc_H, opt_disc, epoch,
                            filename=f"{weights_folder_classe_skipconnections_size_li_osls}/"
                                     f"{config.CHECKPOINT_CRITIC_H}")
            save_checkpoint(disc_Z, opt_disc, epoch,
                            filename=f"{weights_folder_classe_skipconnections_size_li_osls}/"
                                     f"{config.CHECKPOINT_CRITIC_Z}")

        print(f"saving + epoch {epoch} time: {time.time() - start_time_local} s")


if __name__ == "__main__":
    """
    /!\ if rootdirectory isn't cycleGAN/, then it won't work /!\ 
    """
    start_time = time.time()

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

    config.def_transforms()

    main()

    logger(
        f"Total time {time.time() - start_time}",
        True,
        str(time.time() - start_time),
        "total_time"
    )

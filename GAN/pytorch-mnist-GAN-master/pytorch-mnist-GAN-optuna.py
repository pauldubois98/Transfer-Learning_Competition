# Python version
import sys

print(sys.version)

# Imports
import math
import random
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from functools import lru_cache

import optuna
from optuna.trial import TrialState
from optuna.pruners import ThresholdPruner

import time


# Loading data
@lru_cache(maxsize=1)
def define_model(bs):
    print(f'batch_size: {bs}')
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    # the output of torchvision datasets are PILImage images of range [0, 1] and we
    # want data that is centered around 0 with a std of 1 (0.1307 and 0.3081 are the estimated values of the MNIST
    # mean & std)

    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transform, download=True)
    print(f"Size train_dataset: {len(train_dataset)}")
    min_value_dataset = float('inf')
    max_value_dataset = -float('inf')
    for image_idx in range(1000):
        min_value_dataset = min(train_dataset[image_idx][0].min(), min_value_dataset)
        max_value_dataset = max(train_dataset[image_idx][0].max(), max_value_dataset)
    print(f"min value after normalization around 0 with std of 1: {round(min_value_dataset.item(), 2)}")
    print(f"max value after normalization around 0 with std of 1: {round(max_value_dataset.item(), 2)}")

    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transform, download=True)
    print(f"Size test_dataset: {len(test_dataset)}")

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    return min_value_dataset, max_value_dataset, train_loader, train_dataset, test_loader, test_dataset


# Generator
class Generator(nn.Module):
    def __init__(self, g_input_dim, g_output_dim, dropout_generator, double_dropout_generator,
                 min_value_dataset, max_value_dataset, leaky_relu_or_relu):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(g_input_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
        self.dropout_generator = dropout_generator
        self.double_dropout_generator = double_dropout_generator
        self.min_value_dataset = min_value_dataset
        self.max_value_dataset = max_value_dataset
        self.leaky_relu_or_relu = leaky_relu_or_relu

    # Forward method
    def forward(self, x):
        if self.leaky_relu_or_relu == "ReLU":
            x = F.relu(self.fc1(x))
        elif self.leaky_relu_or_relu == "LeakyReLU":
            x = F.leaky_relu(self.fc1(x), 0.2)

        x = F.dropout(x, self.dropout_generator)

        if self.leaky_relu_or_relu == "ReLU":
            x = F.relu(self.fc2(x))
        elif self.leaky_relu_or_relu == "LeakyReLU":
            x = F.leaky_relu(self.fc2(x), 0.2)

        if self.double_dropout_generator:
            x = F.dropout(x, self.dropout_generator)

        if self.leaky_relu_or_relu == "ReLU":
            x = F.relu(self.fc3(x))
        elif self.leaky_relu_or_relu == "LeakyReLU":
            x = F.leaky_relu(self.fc3(x), 0.2)

        x = torch.tanh(self.fc4(x))  # dans [-1; 1]
        x = ((self.min_value_dataset + self.max_value_dataset) / 2) + \
            ((self.max_value_dataset - self.min_value_dataset) / 2) * x

        return x  # dans [-min_value_dataset; max_value_dataset]


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, d_input_dim, dropout_discriminator):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)
        self.dropout_discriminator = dropout_discriminator

    # Forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, self.dropout_discriminator)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, self.dropout_discriminator)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, self.dropout_discriminator)
        return torch.sigmoid(self.fc4(x))  # output dans [0; 1]


# Build network
def build_network(train_dataset, z_dim, dropout_generator, double_dropout_generator, dropout_discriminator,
                  min_value_dataset, max_value_dataset, leaky_relu_or_relu):
    mnist_dim = train_dataset.data.size(1) * train_dataset.data.size(2)
    # drawings dimension, output of the generator and input of the discriminator
    # cf torch_empty_and_tensor_dot_size.py for correct use of the .size method

    print(f"latent space dimension: {z_dim}")
    print(f"drawings dimension: {mnist_dim}")
    print(f"28*28 = {28 * 28}")
    print("\n")

    G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim,
                  dropout_generator=dropout_generator,
                  double_dropout_generator=double_dropout_generator,
                  min_value_dataset=min_value_dataset, max_value_dataset=max_value_dataset,
                  leaky_relu_or_relu=leaky_relu_or_relu).to(device)
    D = Discriminator(d_input_dim=mnist_dim, dropout_discriminator=dropout_discriminator).to(device)

    return G, D, mnist_dim


# Criteria and optimizers
def init_criterion_optimizer(G, D, learning_rate_g, learning_rate_d, beta1):
    # Loss
    criterion = nn.BCELoss()

    # Optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=learning_rate_g, betas=(beta1, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=learning_rate_d, betas=(beta1, 0.999))

    return criterion, G_optimizer, D_optimizer


# Train discriminator
def D_train(G, D, D_optimizer, criterion, threshold_no_train, x, mnist_dim, z_dim, one_sided_label_smoothing):
    """
    Train the discriminator on 2*bs samples, 1*bs real and 1*bs fake data
    :param: x: real data
    :return: discriminator loss for the batch, has the discriminator been trained this batch, was it trained ?,
             x_fake (to train the generator - it'd be computationally heavy to re-generate latent noise
             and re-applying G(z))
    """
    # train discriminator on real data
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(x.size(0), 1) - one_sided_label_smoothing * random.random()
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake data
    z = Variable(torch.randn(x.size(0), z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(x.size(0), 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_real_loss + D_fake_loss) / 2

    to_train = D_loss > threshold_no_train

    if to_train:
        D_loss.backward(retain_graph=True)  # retain_graph because we are going to use x_fake again
        D_optimizer.step()

    return D_loss.data.item(), to_train, x_fake  # the item() method extracts the loss’s value as a Python float


# Train Generator
def G_train(G, G_optimizer, D, criterion, x_fake):
    """
    Train the generator on bs sample
    :param: x_fake: G(z) already calculated in D_train
    """
    y = Variable(torch.ones(x_fake.size(0), 1).to(device))

    D_output = D(x_fake)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()  # the item() method extracts the loss’s value as a Python float


# Test discriminator
def D_val(G, D, criterion, x, mnist_dim, z_dim):
    """
    Test the discriminator on 2*bs samples, 1*bs real and 1*bs fake data
    :param: x: real data
    :return: discriminator validation loss for the batch, has the discriminator been trained this batch, x_fake (to
             train the generator - it'd be computationally heavy to re-generate latent noise and re-applying G(z))
    """
    # train discriminator on real data
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(x.size(0), 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)

    # train discriminator on fake data
    z = Variable(torch.randn(x.size(0), z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(x.size(0), 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)

    # gradient backprop & optimize ONLY D's parameters
    D_loss = (D_real_loss + D_fake_loss) / 2

    return D_loss.data.item(), x_fake  # the item() method extracts the loss’s value as a Python float


# Test Generator
def G_val(G, D, criterion, x_fake):
    """
    Test the generator on bs sample
    :param: x_fake: G(z) already calculated in D_val
    """
    y = Variable(torch.ones(x_fake.size(0), 1).to(device))

    D_output = D(x_fake)
    G_loss = criterion(D_output, y)

    return G_loss.data.item()  # the item() method extracts the loss’s value as a Python float

# Objective
def objective(trial):
    # Hyperparameters
    bs = trial.suggest_int('bs', 16, 1024)
    z_dim = trial.suggest_int('z_dim', 32, 512)
    learning_rate_g = trial.suggest_float('learning_rate_g', 1e-5, 1e-1, log=True)
    learning_rate_d = trial.suggest_float('learning_rate_d', 1e-5, 1e-1, log=True)
    beta1 = trial.suggest_float('beta1', 0.4, 0.95)
    threshold_no_train = trial.suggest_float('threshold_no_train', 1e-10, 0.1, log=True)
    dropout_generator = trial.suggest_float('dropout_generator', 1e-10, 0.5, log=True)
    dropout_discriminator = trial.suggest_float('dropout_discriminator', 1e-10, 0.5, log=True)
    one_sided_label_smoothing = trial.suggest_float('one_sided_label_smoothing', 0., 0.2)
    double_dropout_generator = trial.suggest_categorical('double_dropout_generator', [True, False])
    leaky_relu_or_relu = trial.suggest_categorical('leaky_relu_or_relu', ["LeakyReLU", "ReLU"])

    n_epoch = 300

    # Loading the data
    min_value_dataset, max_value_dataset, train_loader, train_dataset, test_loader, test_dataset = define_model(bs)

    # Build the network
    G, D, mnist_dim = build_network(train_dataset, z_dim,
                                    dropout_generator, double_dropout_generator, dropout_discriminator,
                                    min_value_dataset, max_value_dataset, leaky_relu_or_relu)

    # Criteria & optimizers
    criterion, G_optimizer, D_optimizer = init_criterion_optimizer(G, D, learning_rate_g, learning_rate_d, beta1)

    # End_initialise
    end_initialise = time.time()
    print("Initialisation over:", round(end_initialise - start_time, 2), "s")
    print("\n")

    # Let's go
    for epoch in range(1, n_epoch + 1):
        G.train()
        D_losses, D_was_trained_history, G_losses = [], [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            D.zero_grad()
            G.zero_grad()

            D_loss, D_was_trained, x_fake = D_train(G, D, D_optimizer, criterion, threshold_no_train, x, mnist_dim,
                                                    z_dim, one_sided_label_smoothing)
            D_losses.append(D_loss)
            D_was_trained_history.append(D_was_trained)

            G_losses.append(G_train(G, G_optimizer, D, criterion, x_fake))

        D_was_trained_count = 0
        for booleen in D_was_trained_history:
            if booleen:
                D_was_trained_count += 1

        loss_g = torch.mean(torch.FloatTensor(G_losses))
        loss_d = torch.mean(torch.FloatTensor(D_losses))

        D_val_losses, G_val_losses = [], []
        with torch.no_grad():
            G.eval()
            D.eval()
            for batch_idx, (x, _) in enumerate(test_loader):
                D_val_loss, x_fake = D_val(G, D, criterion, x, mnist_dim, z_dim)
                D_val_losses.append(D_val_loss)

                G_val_loss = G_val(G, D, criterion, x_fake)
                G_val_losses.append(G_val_loss)

        loss_val_g = torch.mean(torch.FloatTensor(G_val_losses))
        loss_val_d = torch.mean(torch.FloatTensor(D_val_losses))

        print('[%d/%d]: loss_d (training): %.3f, loss_d (validation): %.3f, loss_g (training): %.3f, '
              'loss_g (validation): %.3f,  discriminator trained: %d/%d, average epoch time: %.2f s'
              % (epoch, n_epoch, loss_d, loss_val_d,
                 loss_g, loss_val_g, D_was_trained_count, math.ceil(len(train_dataset) / bs),
                 (time.time() - end_initialise) / epoch))

        trial.report(loss_val_g + loss_val_d, epoch)

    return loss_val_g + loss_val_d


if __name__ == "__main__":
    start_time = time.time()

    # Device configuration
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)
    print("\n")

    # Optuna
    study = optuna.create_study(direction="minimize", pruner=ThresholdPruner(upper=15.0, n_warmup_steps=15))
    study.optimize(objective, n_trials=None, timeout=6 * 3600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("Saving ...")
    joblib.dump(study, "study.pkl")
    print("Saving done")

    print("\n")
    print("Total time:", round(time.time() - start_time, 2), "s")

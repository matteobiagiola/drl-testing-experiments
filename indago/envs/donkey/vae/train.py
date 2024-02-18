"""
MIT License

Copyright (c) 2018 Roma Sokolkov
Copyright (c) 2018 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import argparse
import base64
import logging
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from stable_baselines3.common.utils import set_random_seed
from tqdm import tqdm

from indago.config import CAMERA_HEIGHT, CAMERA_WIDTH
from indago.envs.donkey.scenes.simulator_scenes import SIMULATOR_SCENES_DICT
from indago.envs.donkey.vae.data_loader import DataLoader
from indago.envs.donkey.vae.vae import VAE
from indago.utils.torch_utils import DEVICE, from_numpy
from log import Log

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--z-size", help="Latent space", type=int, default=64)
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    parser.add_argument(
        "--n-samples", help="Max number of samples", type=int, default=-1
    )
    parser.add_argument(
        "--num-workers", help="Num workers for data loader", type=int, default=2
    )
    parser.add_argument("--batch-size", help="Batch size", type=int, default=64)
    parser.add_argument(
        "--learning-rate", help="Learning rate", type=float, default=1e-4
    )
    parser.add_argument(
        "--kl-tolerance", help="KL tolerance (to cap KL loss)", type=float, default=0.5
    )
    parser.add_argument("--n-epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--verbose", help="Verbosity", type=int, default=0)
    parser.add_argument("--augment", help="Data augmentation", action="store_true")

    args = parser.parse_args()
    logger = Log("vae_train")

    simulator_scene = SIMULATOR_SCENES_DICT["generated_track"]

    assert os.path.exists(
        os.path.join("logs", "{}".format(simulator_scene.get_scene_name()))
    ), "{} does not exist".format(
        os.path.join("logs", "{}".format(simulator_scene.get_scene_name()))
    )

    logging.basicConfig(
        filename=os.path.join(
            os.path.join(
                "logs", "{}".format(simulator_scene.get_scene_name(), args.z_size)
            ),
            "vae-train-{}.txt".format(args.z_size),
        ),
        filemode="w",
        level=logging.DEBUG,
    )

    if args.seed == -1:
        args.seed = np.random.randint(2**32 - 1)

    set_random_seed(args.seed)

    logger.info("Args: {}".format(args))
    if args.expert:
        expert_dataset_filepath = os.path.join(
            "logs", "{}".format(simulator_scene.get_scene_name()), "expert_dataset.npz"
        )
        assert os.path.exists(expert_dataset_filepath), "{} does not exist".format(
            expert_dataset_filepath
        )

        expert_dataset = np.load(expert_dataset_filepath)

        image_path = False

        observations = expert_dataset["obs"]
        images = np.zeros(shape=(len(observations), *(CAMERA_HEIGHT, CAMERA_WIDTH, 3)))
        for idx, observation_string in enumerate(observations):
            image = Image.open(BytesIO(base64.b64decode(observation_string)))
            image_array = np.array(image, dtype=np.float32)
            images[idx] = image_array
    else:
        images_filepath = os.path.join(
            "logs", "{}".format(simulator_scene.get_scene_name()), "vae_images"
        )
        assert os.path.exists(images_filepath), "{} does not exist".format(
            images_filepath
        )

        images = []
        for image_path in os.listdir(images_filepath):
            if ".jpg" in image_path:
                images.append(os.path.join(images_filepath, image_path))
                if args.n_samples == len(images):
                    break

        images = np.asarray(images)
        image_path = True

    n_samples = len(images)
    np.random.shuffle(images)

    if args.n_samples > 0:
        n_samples = min(n_samples, args.n_samples)

    logger.info("Num images: {}".format(n_samples))

    validation_split = 0.02

    split = int(np.floor(validation_split * n_samples))
    indices = list(range(n_samples))
    np.random.shuffle(indices)

    indices_train, indices_validation = indices[split:], indices[:split]

    # split indices into minibatches. minibatchlist is a list of lists; each
    # list is the id of the observation preserved through the training
    minibatchlist_train = [
        np.array(sorted(indices_train[start_idx : start_idx + args.batch_size]))
        for start_idx in range(
            0, len(indices_train) - args.batch_size + 1, args.batch_size
        )
    ]
    minibatchlist_validation = [
        np.array(sorted(indices_validation[start_idx : start_idx + args.batch_size]))
        for start_idx in range(
            0, len(indices_validation) - args.batch_size + 1, args.batch_size
        )
    ]

    data_loader_train = DataLoader(
        minibatchlist_train,
        images,
        n_workers=args.num_workers,
        image_path=image_path,
        augment=args.augment,
    )
    data_loader_validation = DataLoader(
        minibatchlist_validation,
        images,
        n_workers=args.num_workers,
        image_path=image_path,
        augment=args.augment,
    )

    save_path = os.path.join(
        "logs",
        "{}".format(simulator_scene.get_scene_name()),
        "vae-{}.pkl".format(args.z_size),
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    vae = VAE(in_channels=3, latent_dim=args.z_size).to(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    best_val_loss = np.inf

    train_losses = []
    val_losses = []

    for epoch in range(args.n_epochs):
        progress_bar = tqdm(total=len(minibatchlist_train))
        if epoch > 0:  # test untrained net first
            vae.train()
            train_loss = 0
            for obs, target_obs in data_loader_train:
                obs = from_numpy(obs)
                # ===================forward=====================
                obs_predicted, mu, logvar = vae.forward(input=obs)
                loss = vae.loss_function(obs_predicted, obs, mu, logvar)["loss"]
                train_loss += loss.item()
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
            train_loss /= len(indices_train)
            train_losses.append(train_loss)
            logger.info("{}/{} Train loss: {}".format(epoch, args.n_epochs, train_loss))

            if epoch % 2 == 0:
                # calculate accuracy on validation set
                val_loss = 0
                means, logvars, target_obss = list(), list(), list()
                with torch.no_grad():
                    # switch model to evaluation mode
                    vae.eval()
                    test_loss = 0
                    for obs, target_obs in data_loader_validation:
                        obs = from_numpy(obs)
                        # ===================forward=====================
                        obs_predicted, mu, logvar = vae.forward(input=obs)
                        loss = vae.loss_function(obs_predicted, obs, mu, logvar)["loss"]
                        val_loss += loss.item()

                val_loss /= len(indices_validation)
                val_losses.append(val_loss)
                logger.info(
                    "{}/{} Validation loss: {}".format(epoch, args.n_epochs, val_loss)
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info(
                        "New best loss: {}. Saving VAE to path: {}".format(
                            best_val_loss, save_path
                        )
                    )
                    vae.save(filepath=save_path)
            else:
                if len(val_losses) > 0:
                    val_losses.append(val_losses[-1])
                else:
                    val_losses.append(train_losses[0])

        progress_bar.close()

    plt.figure()
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("# Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(
        os.path.join(
            "logs",
            "{}".format(simulator_scene.get_scene_name()),
            "training-loss-{}.pdf".format(args.z_size),
        ),
        format="pdf",
    )

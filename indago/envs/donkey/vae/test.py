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
import os
import random

import cv2
import numpy as np
import torch

from indago.envs.donkey.vae.data_loader import preprocess_raw_image
from indago.envs.donkey.vae.vae import VAE
from indago.utils.torch_utils import from_numpy_no_device, to_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="Folder with real images", type=str, required=True)
parser.add_argument(
    "-vae", "--vae-path", help="Path to saved VAE", type=str, default=""
)
parser.add_argument(
    "--n-samples", help="Max number of samples to process", type=int, default=-1
)
parser.add_argument(
    "--z-size",
    help="Latent space. Should match the latent space of the trained vae",
    type=int,
    default=64,
)
parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
parser.add_argument(
    "--show-images", help="Show images on screen", action="store_true", default=False
)
parser.add_argument(
    "--rank-by-loss",
    help="Save images on disk ranked by their reconstruction loss",
    action="store_true",
    default=False,
)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)

images = [
    os.path.join(args.folder, f)
    for f in os.listdir(args.folder)
    if f.endswith(".jpg") or f.endswith(".png")
]

n_samples = len(images)
np.random.shuffle(images)

vae = VAE(in_channels=3, latent_dim=args.z_size)
vae.load(args.vae_path)

range_fn = range(args.n_samples) if args.n_samples > 0 else range(n_samples)
losses = []

if args.rank_by_loss:
    assert not args.show_images, "show_images cannot be True when rank_by_loss is True"
    assert (
        args.folder is not None
    ), "folder arg should not be None when rank_by_loss is True"
    os.makedirs(os.path.join(args.folder, "rank"), exist_ok=True)

for i in range_fn:
    # Load test image
    image_idx = np.random.randint(n_samples)
    if args.folder is None:
        image = images[image_idx]
    else:
        image = cv2.imread(images[image_idx])

    # image = get_image_sample(image_path=images[image_idx], reshape=False)
    # image_input = get_image_sample(image_path=images[image_idx])
    preprocessed_image = preprocess_raw_image(image=image, reshape=True)
    image_tensor = from_numpy_no_device(preprocessed_image)
    reconstructed_image = vae.generate(x=image_tensor)
    reconstructed_image_np = to_numpy(tensor=reconstructed_image)
    # encoded = vae.encode_from_raw_image(input_image)
    # reconstructed_image = vae.decode(encoded)[0]
    with torch.no_grad():
        obs_predicted, mu, logvar = vae.forward(input=image_tensor)
        loss = vae.loss_function(obs_predicted, image_tensor, mu, logvar)["loss"]
        losses.append(loss)
        if args.show_images:
            print("Loss: {}".format(loss))

        if args.rank_by_loss:
            index_image = images[image_idx][
                images[image_idx].rindex(os.sep) + 1 : images[image_idx].index(".")
            ]
            cv2.imwrite(
                filename=os.path.join(
                    args.folder, "rank", "l_{}_i_{}.png".format(int(loss), index_image)
                ),
                img=image,
            )

    # Plot reconstruction
    if args.show_images:
        if args.folder is None:
            cv2.imshow(
                "Original", preprocessed_image.reshape(preprocessed_image.shape[1:])
            )
        else:
            cv2.imshow("Original", image)

    if args.show_images:
        cv2.imshow(
            "Reconstruction",
            reconstructed_image_np.reshape(reconstructed_image_np.shape[1:]),
        )

    if args.show_images:
        cv2.waitKey(0)

    if args.n_samples > 0 and args.folder is not None:
        if args.n_samples == i:
            break

print(
    "Loss stats: Mean - {}, Std - {}, Max - {}, Min - {}".format(
        np.mean(losses), np.std(losses), np.max(losses), np.min(losses)
    )
)

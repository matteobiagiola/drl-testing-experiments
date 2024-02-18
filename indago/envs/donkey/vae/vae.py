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

# author: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import abc
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from indago.envs.donkey.vae.data_loader import preprocess_image, preprocess_raw_image
from indago.utils.torch_utils import DEVICE, from_numpy, from_numpy_no_device, to_numpy


class PreProcessImage(nn.Module):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class PostProcessImage(nn.Module):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class VAE(nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        kl_tolerance: float = 0.5,
        hidden_dims: List = None,
        beta: float = 1.0,
        normalization_mode: str = "rl",
    ) -> None:
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.z_size = latent_dim  # for compatibility with TF VAE and Recorder class
        self.beta = beta
        self.kl_tolerance = kl_tolerance
        self.normalization_mode = normalization_mode

        self.encoder = nn.Sequential(
            PreProcessImage(),
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=4, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            PostProcessImage(),
        )

        # hardcoded based on input dimensions and encoder cnn layers otherwise
        # issues with gpu (see how stable baselines 3 solved the issue)
        # (3 x 8 x 265), 3 = input channels, 8 = resulting from convolution, 256 = output of last layer of cnn
        n_flatten = 6144

        self.fc_mu = nn.Linear(n_flatten, latent_dim)
        self.fc_var = nn.Linear(n_flatten, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, n_flatten)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=5, stride=2
            ),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=in_channels, kernel_size=4, stride=2
            ),
            PostProcessImage(),
            nn.Sigmoid(),
        )

    def encode_from_raw_image(self, raw_image):
        """
        :param raw_image: (np.ndarray) BGR image
        """

        preprocessed_image = preprocess_raw_image(image=raw_image)
        image_tensor = from_numpy(preprocessed_image)
        return to_numpy(self.encode(input=image_tensor)[0])

    def get_latent_from_raw_image(self, raw_image) -> Tensor:
        preprocessed_image = preprocess_raw_image(image=raw_image)
        image_tensor = from_numpy(preprocessed_image)
        return self.get_latent(input=image_tensor, training=False)

    def encode(self, input: Tensor) -> Tuple:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def get_latent(self, input: Tensor, training: bool = True) -> Tensor:
        mu, log_var = self.encode(input)
        if training:
            return reparameterize(mu, log_var)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # hardcoded based on encoder cnn layers (see n_flatten variable: transpose of 3 x 8 x 256)
        result = result.view(-1, 256, 3, 8)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def get_reconstruction_and_loss(
        self, observation: np.ndarray, roi: bool, convert_to_rgb: bool = False
    ) -> Tuple[np.ndarray, float]:

        with torch.no_grad():
            observation = preprocess_image(
                image=observation, roi=roi, convert_to_rgb=convert_to_rgb
            )
            # r = ROI
            # image = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # Convert RGB to BGR
            # image = image[:, :, ::-1]
            # self.image_array = image
            observation = observation.reshape((1,) + observation.shape)
            image_tensor = from_numpy_no_device(observation)
            mu, log_var = self.encode(image_tensor)
            z = reparameterize(mu, log_var)

            encoded_image = z

            obs_predicted = self.decode(z=encoded_image)
            loss = self.loss_function(obs_predicted, image_tensor, mu, log_var)["loss"]
            reconstructed_image_np = to_numpy(tensor=obs_predicted)

        return reconstructed_image_np, loss.item()

    def forward(self, input: Tensor, training: bool = True, **kwargs) -> Tuple:
        mu, log_var = self.encode(input)
        if training:
            z = reparameterize(mu, log_var)
        else:
            z = mu
        return self.decode(z), mu, log_var

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input, reduction="sum")

        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1)
        if self.kl_tolerance > 0:
            scalar = torch.FloatTensor([self.kl_tolerance * self.latent_dim]).to(DEVICE)
            kld_loss = torch.maximum(kld_loss, scalar.expand_as(kld_loss))
        kld_loss = torch.mean(kld_loss)

        # loss = recons_loss + kld_weight * kld_loss
        loss = recons_loss + self.beta * kld_loss

        return {"loss": loss, "Reconstruction_Loss": recons_loss, "KLD": -kld_loss}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, training: bool = True, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, training=training)[0]

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath, load_on_device: bool = False) -> nn.Module:
        if load_on_device:
            self.load_state_dict(
                torch.load(filepath, map_location=torch.device(DEVICE))
            )
        else:
            self.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
        self.eval()
        return self

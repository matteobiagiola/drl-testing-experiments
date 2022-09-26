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

# Original code from https://github.com/araffin/robotics-rl-srl
# Authors: Antonin Raffin, René Traoré, Ashley Hill
import queue
import random
import time
from multiprocessing import Process, Queue

import cv2
import imgaug.augmenters as iaa
import numpy as np
import torchvision.transforms.functional as vision_fn
from imgaug.augmenters import Sometimes
from joblib import Parallel, delayed
from PIL import Image, ImageChops

from indago.config import IMAGE_HEIGHT, IMAGE_WIDTH, ROI


def get_image_sample(image_path: str, reshape: bool = True) -> np.ndarray:
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError("tried to load {}.jpg, but it was not found".format(image_path))

    return preprocess_raw_image(image=im, reshape=reshape)


def preprocess_raw_image(image, reshape: bool = True) -> np.ndarray:
    target_img = preprocess_image(image)
    if reshape:
        target_img = target_img.reshape((1,) + target_img.shape)
    return target_img


def postprocess_output(output: np.ndarray) -> np.ndarray:
    output = output.reshape(output.shape[1:])
    return denormalize(x=output, mode="rl")


def preprocess_input(x, mode="rl"):
    """
    Normalize input.

    :param x: (np.ndarray) (RGB image with values between [0, 255])
    :param mode: (str) One of "image_net", "tf" or "rl".
        - rl: divide by 255 only (rescale to [0, 1])
        - image_net: will zero-center each color channel with
            respect to the ImageNet dataset,
            with scaling.
            cf http://pytorch.org/docs/master/torchvision/models.html
        - tf: will scale pixels between -1 and 1,
            sample-wise.
    :return: (np.ndarray)
    """
    # RL mode: divide only by 255
    x /= 255.0

    if mode == "tf":
        x -= 0.5
        x *= 2.0
    elif mode == "image_net":
        assert x.shape[-1] == 3, "Color channel must be at the end of the tensor {}".format(x.shape)
        # Zero-center by mean pixel
        x[..., 0] -= 0.485
        x[..., 1] -= 0.456
        x[..., 2] -= 0.406
        # Scaling
        x[..., 0] /= 0.229
        x[..., 1] /= 0.224
        x[..., 2] /= 0.225
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for preprocessing")
    return x


def denormalize(x, mode="rl"):
    """
    De normalize data (transform input to [0, 1])

    :param x: (np.ndarray)
    :param mode: (str) One of "image_net", "tf", "rl".
    :return: (np.ndarray)
    """

    if mode == "tf":
        x /= 2.0
        x += 0.5
    elif mode == "image_net":
        # Scaling
        x[..., 0] *= 0.229
        x[..., 1] *= 0.224
        x[..., 2] *= 0.225
        # Undo Zero-center
        x[..., 0] += 0.485
        x[..., 1] += 0.456
        x[..., 2] += 0.406
    elif mode == "rl":
        pass
    else:
        raise ValueError("Unknown mode for denormalize")
    # Clip to fix numeric imprecision (1e-09 = 0)
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def preprocess_image(image, convert_to_rgb=False, normalize=True, roi: bool = True):
    """
    Crop, resize and normalize image.
    Optionally it also converts the image from BGR to RGB.

    :param image: (np.ndarray) image (BGR or RGB)
    :param convert_to_rgb: (bool) whether the conversion to rgb is needed or not
    :param normalize: (bool) Whether to normalize or not
    :return: (np.ndarray)
    """
    # Crop
    # Region of interest
    im = image
    if roi:
        r = ROI
        image = image[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
        # Resize
        im = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    # im = np.moveaxis(im, 2, 0)
    # Convert BGR to RGB
    if convert_to_rgb:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Normalize
    if normalize:
        im = preprocess_input(im.astype(np.float32), mode="rl")

    return im


def get_image_augmenter():
    """
    :return: (iaa.Sequential) Image Augmenter
    """

    return iaa.Sequential(
        [
            Sometimes(0.5, iaa.Fliplr(1)),
            # TODO: add shadows, see: https://markku.ai/post/data-augmentation/
            # Add shadows (from https://github.com/OsamaMazhar/Random-Shadows-Highlights)
            Sometimes(0.3, RandomShadows(1.0)),
            Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 2.0))),
            Sometimes(0.5, iaa.MotionBlur(k=(3, 11), angle=(0, 360))),
            Sometimes(0.4, iaa.Add((-25, 25), per_channel=0.5)),
            # 20% of the corresponding size of the height and width
            Sometimes(0.3, iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
        ],
        random_order=True,
    )


# Adapted from https://github.com/OsamaMazhar/Random-Shadows-Highlights
class RandomShadows(iaa.meta.Augmenter):
    def __init__(
        self,
        p=0.5,
        high_ratio=(1, 2),
        low_ratio=(0.01, 0.5),
        left_low_ratio=(0.4, 0.6),
        left_high_ratio=(0, 0.2),
        right_low_ratio=(0.4, 0.6),
        right_high_ratio=(0, 0.2),
        seed=None,
        name=None,
    ):
        super(RandomShadows, self).__init__(seed=seed, name=name)

        self.p = p
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.left_low_ratio = left_low_ratio
        self.left_high_ratio = left_high_ratio
        self.right_low_ratio = right_low_ratio
        self.right_high_ratio = right_high_ratio

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i in range(batch.nb_rows):
            if random.uniform(0, 1) < self.p:
                batch.images[i] = self.process(
                    batch.images[i],
                    self.high_ratio,
                    self.low_ratio,
                    self.left_low_ratio,
                    self.left_high_ratio,
                    self.right_low_ratio,
                    self.right_high_ratio,
                )
        return batch

    @staticmethod
    def process(
        img, high_ratio, low_ratio, left_low_ratio, left_high_ratio, right_low_ratio, right_high_ratio,
    ):

        img = Image.fromarray(img)
        w, h = img.size
        # h, w, c = img.shape
        high_bright_factor = random.uniform(high_ratio[0], high_ratio[1])
        low_bright_factor = random.uniform(low_ratio[0], low_ratio[1])

        left_low_factor = random.uniform(left_low_ratio[0] * h, left_low_ratio[1] * h)
        left_high_factor = random.uniform(left_high_ratio[0] * h, left_high_ratio[1] * h)
        right_low_factor = random.uniform(right_low_ratio[0] * h, right_low_ratio[1] * h)
        right_high_factor = random.uniform(right_high_ratio[0] * h, right_high_ratio[1] * h)

        tl = (0, left_high_factor)
        bl = (0, left_high_factor + left_low_factor)

        tr = (w, right_high_factor)
        br = (w, right_high_factor + right_low_factor)

        contour = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros([h, w, 3], np.uint8)
        cv2.fillPoly(mask, [contour], (255, 255, 255))
        inverted_mask = cv2.bitwise_not(mask)
        # we need to convert this cv2 masks to PIL images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # we skip the above convertion because our mask is just black and white
        mask_pil = Image.fromarray(mask)
        inverted_mask_pil = Image.fromarray(inverted_mask)

        low_brightness = vision_fn.adjust_brightness(img, low_bright_factor)
        low_brightness_masked = ImageChops.multiply(low_brightness, mask_pil)
        high_brightness = vision_fn.adjust_brightness(img, high_bright_factor)
        high_brightness_masked = ImageChops.multiply(high_brightness, inverted_mask_pil)

        return np.array(ImageChops.add(low_brightness_masked, high_brightness_masked))

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return []


class DataLoader(object):
    def __init__(
        self,
        minibatchlist,
        images: np.ndarray,
        n_workers=1,
        image_path: bool = False,
        infinite_loop=True,
        max_queue_len=4,
        is_training=False,
        augment=False,
    ):
        """
        A Custom dataloader to preprocessing images and feed them to the network.

        :param minibatchlist: ([np.array]) list of observations indices (grouped per minibatch)
        :param images: (np.array) Array of path to images
        :param n_workers: (int) number of preprocessing worker (load and preprocess each image)
        :param infinite_loop: (bool) whether to have an iterator that can be resetted, set to False, it
        :param max_queue_len: (int) Max number of minibatches that can be preprocessed at the same time
        :param is_training: (bool)
        :param augment: (bool) Whether to use image augmentation or not
        """
        super(DataLoader, self).__init__()
        self.n_workers = n_workers
        self.infinite_loop = infinite_loop
        self.n_minibatches = len(minibatchlist)
        self.minibatchlist = minibatchlist
        self.images = images
        self.image_path = image_path
        self.shuffle = is_training
        self.queue = Queue(max_queue_len)
        self.process = None
        self.augmenter = None
        if augment:
            self.augmenter = get_image_augmenter()
        self.start_process()

    @staticmethod
    def create_minibatch_list(n_samples, batch_size):
        """
        Create list of minibatches.

        :param n_samples: (int)
        :param batch_size: (int)
        :return: ([np.array])
        """
        minibatchlist = []
        for i in range(n_samples // batch_size + 1):
            start_idx = i * batch_size
            end_idx = min(n_samples, (i + 1) * batch_size)
            minibatchlist.append(np.arange(start_idx, end_idx))
        return minibatchlist

    def start_process(self):
        """Start preprocessing process"""
        self.process = Process(target=self._run)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def _run(self):
        start = True
        with Parallel(n_jobs=self.n_workers, batch_size="auto", backend="threading") as parallel:
            while start or self.infinite_loop:
                start = False

                if self.shuffle:
                    indices = np.random.permutation(self.n_minibatches).astype(np.int64)
                else:
                    indices = np.arange(len(self.minibatchlist), dtype=np.int64)

                for minibatch_idx in indices:

                    images = self.images[self.minibatchlist[minibatch_idx]]

                    if self.n_workers <= 1:
                        batch = [self._make_batch_element(image, self.augmenter, self.image_path) for image in images]

                    else:
                        batch = parallel(
                            delayed(self._make_batch_element)(image, self.augmenter, self.image_path) for image in images
                        )

                    batch_input = np.concatenate([batch_elem[0] for batch_elem in batch], axis=0)
                    batch_target = np.concatenate([batch_elem[1] for batch_elem in batch], axis=0)

                    if self.shuffle:
                        self.queue.put((minibatch_idx, batch_input, batch_target))
                    else:
                        self.queue.put((batch_input, batch_target))

                    # Free memory
                    del batch_input, batch_target, batch

                self.queue.put(None)

    @classmethod
    def _make_batch_element(cls, image, augmenter=None, image_path: bool = True):
        """
        :param image: (str) path to an image
        :param augment: (iaa.Sequential) Image augmenter
        :return: (np.ndarray, np.ndarray)
        """
        if image_path:
            im = cv2.imread(image)
            if im is None:
                raise ValueError("tried to load {}.jpg, but it was not found".format(image))
        else:
            im = image

        target_img = preprocess_image(im)
        target_img = target_img.reshape((1,) + target_img.shape)

        if augmenter is not None:
            preprocessed_image = preprocess_image(im.astype(np.float32), normalize=False)
            input_img = augmenter.augment_image(preprocessed_image)
            # Normalize
            input_img = preprocess_input(input_img.astype(np.float32), mode="rl")
            input_img = input_img.reshape((1,) + input_img.shape)
        else:
            input_img = target_img.copy()

        return input_img, target_img

    def __len__(self):
        return self.n_minibatches

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                val = self.queue.get_nowait()
                break
            except queue.Empty:
                time.sleep(0.001)
                continue
        if val is None:
            raise StopIteration
        return val

    def __del__(self):
        if self.process is not None:
            self.process.terminate()

import os
import sys
from itertools import islice

import torch
from PIL import Image
import clip
from tqdm import tqdm

from vector_weight_learning import fvecs_converter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root


def get_encoder(id: int):
    if id == 0:
        return ClipEncoderImage()
    elif id == 1:
        return ClipEncoderText()
    elif id == 2:
        return ClipEncoderImageText()
    elif id == 3:
        return Clip4CirEncoderImageText()


class EncoderBase(torch.nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()

    # open meta file, in which each line is a path of image
    def batch(self, dataset, size):
        raise NotImplementedError

    def encode(self, path, dataset, size, flag):
        raise NotImplementedError


class ClipEncoderImage(EncoderBase):
    def __init__(self):
        super().__init__()
        self.modal, self.transform = clip.load('ViT-B/32', device=device)

    def batch(self, dataset, size):
        with open(dataset[0], 'r') as file:
            while True:
                lines = list(islice(file, size))
                if not lines:
                    break

                data = [self.transform(Image.open(path.rstrip('\n'))).to(device) for path in lines]
                yield torch.stack(data)

    def encode(self, path, dataset, size, flag=True):
        features = []
        with open(dataset[0], 'r') as file:
            total = len(list(file))
        if not flag:
            features = torch.zeros(1, 512)
            fvecs_converter.to_fvecs(path, features)
            return
        for data in tqdm(self.batch(dataset=dataset, size=size), total=total // size, desc="[CLIP - Image]"):
            with torch.no_grad():
                image = self.modal.encode_image(data)
                features.extend(image)
        fvecs_converter.to_fvecs(path, features)


class ClipEncoderText(EncoderBase):
    def __init__(self):
        super().__init__()
        self.modal, self.transform = clip.load('ViT-B/32', device=device)

    def batch(self, dataset, size):
        with open(dataset[0], 'r') as file:
            while True:
                lines = list(islice(file, size))
                if not lines:
                    break

                data = [line.rstrip('\n') for line in lines]
                yield data

    def encode(self, path, dataset, size, flag=True):
        features = []
        with open(dataset[0], 'r') as file:
            total = len(list(file))
        if not flag:
            features = torch.zeros(1, 512)
            fvecs_converter.to_fvecs(path, features)
            return
        for data in tqdm(self.batch(dataset=dataset, size=size), total=total // size, desc="[CLIP - Text]"):
            with torch.no_grad():
                text = self.modal.encode_text(clip.tokenize(data).to(device))
                features.extend(text)
        fvecs_converter.to_fvecs(path, features)


class ClipEncoderImageText(EncoderBase):
    def __init__(self):
        super().__init__()
        self.modal, self.transform = clip.load('ViT-B/32', device=device)

    def batch(self, dataset, size):
        with open(dataset[0], 'r') as image_file, open(dataset[1], 'r') as text_file:
            while True:
                image_lines = list(islice(image_file, size))
                text_lines = list(islice(text_file, size))
                if not image_lines and not text_lines:
                    break

                image_data = [self.transform(Image.open(path.rstrip('\n'))).to(device) for path in image_lines]
                text_data = [text.rstrip('\n') for text in text_lines]
                if len(image_data) != 0:
                    image_data = torch.stack(image_data)
                yield image_data, text_data

    def encode(self, path, dataset, size, flag=True):
        features = []
        with open(dataset[0], 'r') as image_file:
            total = len(list(image_file))
        for image_data, text_data in tqdm(self.batch(dataset=dataset, size=size), total=total // size,
                                          desc="[CLIP - Image + Text]"):
            with torch.no_grad():
                image = self.modal.encode_image(image_data) if len(image_data) != 0 else []
                text = self.modal.encode_text(clip.tokenize(text_data).to(device)) if len(text_data) != 0 else []
                max_len = max(len(image), len(text))
                if len(image) < max_len:
                    image = torch.zeros((max_len - len(image), 512))
                if len(text) < max_len:
                    text = torch.zeros((max_len - len(text), 512))
                combine = torch.cat([image, text], dim=1)
                features.extend(combine)
        fvecs_converter.to_fvecs(path, features)


class Clip4CirEncoderImageText(EncoderBase):
    def __init__(self):
        super().__init__()
        self.model, self.clip_preprocess = clip.load("RN50x4")
        self.model.eval().to(device)
        self.input_dim = self.model.visual.input_resolution
        self.feature_dim = self.model.visual.output_dim
        from pojo.clip4cir.data_utils import targetpad_transform
        self.preprocess = targetpad_transform(1.25, self.input_dim)

    def batch(self, dataset, size):
        with open(dataset[0], 'r') as image_file, open(dataset[1], 'r') as text_file:
            while True:
                image_lines = list(islice(image_file, size))
                text_lines = list(islice(text_file, size))
                if not image_lines and not text_lines:
                    break

                image_data = []
                for image_path in image_lines:
                    image_path = image_path.strip('\n')
                    pil_image = Image.open(image_path).convert('RGB')
                    from pojo.clip4cir.data_utils import targetpad_transform
                    image = targetpad_transform(1.25, self.model.visual.input_resolution)(pil_image).to(device)
                    image_data.append(image)
                text_data = [text.rstrip('\n') for text in text_lines]

                if len(image_data) != 0:
                    image_data = torch.stack(image_data)
                yield image_data, text_data

    def encode(self, path, dataset, size, flag=True):
        features = []
        with open(dataset[0], 'r') as file:
            total = len(list(file))
        data_type = torch.float16 if torch.cuda.is_available() else torch.float32
        combiner = torch.hub.load(os.path.join(root, 'pojo', 'clip4cir'), source='local', model='combiner',
                                  dataset='cirr')
        combiner = torch.jit.script(combiner).type(data_type).to(device).eval()

        for image_data, text_data in tqdm(self.batch(dataset=dataset, size=size), total=total // size,
                                          desc="CLIP4CIR - Image + Text"):
            with torch.no_grad():
                image = self.model.encode_image(image_data) if len(image_data) != 0 else []
                text = self.model.encode_text(clip.tokenize(text_data).to(device)) if len(text_data) != 0 else []
                max_len = max(len(image), len(text))
                if len(image) < max_len:
                    image = torch.zeros((max_len - len(image), 640))
                if len(text) < max_len:
                    text = torch.zeros((max_len - len(text), 640))
                combine = combiner.combine_features(image, text)
                features.extend(combine)
        fvecs_converter.to_fvecs(path, features)

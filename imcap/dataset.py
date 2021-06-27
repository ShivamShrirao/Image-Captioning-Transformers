import torch
import torchtext
import torchvision

from torch.utils.data import DataLoader
from torchvision import datasets, transforms as T
from torchvision.io import read_file, decode_jpeg, ImageReadMode
from torchtext.vocab import build_vocab_from_iterator

import os
import numpy as np
from tqdm import tqdm


input_size = 224


preproc = {
    'train': T.Compose([
        T.RandomResizedCrop(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(input_size),
        lambda image: image.convert("RGB"),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': T.Compose([
        T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(input_size),
        lambda image: image.convert("RGB"),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class TensorCocoCaptions(datasets.CocoDetection):
    def __getitem__(self, index: int):
        return super().__getitem__(index % len(self.ids))
    
    def _load_target(self, id):
        return self.tokens_dict[id]
    
    def fill_token_dict(self, tokenizer, vocab, bos_idx, eos_idx):
        self.tokens_dict = {}                       # To save preprocessed captions as tokens.
        for id in tqdm(self.ids):
            captions = self._load_caption(id)
            self.tokens_dict[id] = [torch.tensor([bos_idx] + vocab(tokenizer(cap)) + [eos_idx]) #, dtype=torch.int32)
                                    for cap in captions]
    
    def _load_caption(self, id):
        return [ann["caption"] for ann in super()._load_target(id)]

    def _load_image(self, id):
        path = self.coco.loadImgs(id)[0]["file_name"]
        data = read_file(os.path.join(self.root, path))
        return data
        # return decode_jpeg(data, ImageReadMode.RGB)#, device=DEVICE)
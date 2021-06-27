from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

from torch.nn.utils.rnn import pad_sequence

from random import shuffle
from random import randint


class ExternalInputIterator(object):
    def __init__(self, dataset, batch_size, pad_idx, training=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pad_idx = pad_idx
        self.training = training
        if self.training: shuffle(self.dataset.ids)

    def __iter__(self):
        self.idx = 0
        if self.training: shuffle(self.dataset.ids)
        return self

    def __next__(self):
        img_batch = []
        cap_batch = []

        if self.idx >= len(self.dataset):
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            img, caps = self.dataset[self.idx]
            img_batch.append(img)
            cap = caps[randint(0,len(caps)-1) if self.training else 0]
            cap_batch.append(cap)
            self.idx += 1
        cap_batch = pad_sequence(cap_batch, batch_first=True, padding_value=self.pad_idx)#.type(torch.long)
        return (img_batch, cap_batch)

    def __len__(self):
        return len(self.dataset)

    next = __next__


def ExternalSourcePipeline(batch_size, num_threads, device_id, external_data, input_size, training=True):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        images, labels = fn.external_source(source=external_data, num_outputs=2)
        if training:
            images = fn.decoders.image_random_crop(images, device='mixed', output_type=types.RGB, num_attempts=100, memory_stats=True)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(images, device='mixed', output_type=types.RGB)
            mirror = False
        images = fn.resize(images, device='gpu', resize_shorter=input_size, interp_type=types.INTERP_TRIANGULAR)
        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(input_size, input_size),
                                          mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                          std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                          mirror=mirror)
        labels = labels.gpu()
        pipe.set_outputs(images, labels)
    return pipe
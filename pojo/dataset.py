import os
import random
import sys
from os import listdir
import torch.utils.data
import torchvision

root = os.getcwd()
if sys.platform.startswith('win'):
    root = '\\\\?\\' + root
query_path = os.path.join(root, 'dataset', 'query')
meta_path = os.path.join(root, 'dataset', 'meta')

# the name is to distinguish different modality in same dataset
# they are also used in saving search
# the Modal has type and name. later will create new class called Modal
modal_name = ['image', 'text']


def get_modal_type(id):
    modal_type = ['image', 'text']
    return modal_type[id]


random.seed(42)


def create_meta(id):
    if int(id) < 2:
        MitStates(
            path=os.path.join(root, 'dataset', 'MitStates'),
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ])
        ).create_meta()
        MitStates.generate_query_pair()
    elif id < 10:
        # do sth with other dataset
        return


class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset"""

    def __init__(self):
        super(BaseDataset, self).__init__()

    def get_loader(self, batch_size, shuffle=False, drop_last=False, num_workers=0):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                           drop_last=drop_last, collate_fn=lambda i: i)


class MitStates(BaseDataset):
    def __init__(self, path, transform=None):
        super(MitStates, self).__init__()
        self.path = path
        self.transform = transform

    def create_meta(self):
        image_path = os.path.join(meta_path, '0.txt')
        text_path = os.path.join(meta_path, '1.txt')
        with open(image_path, 'w') as image_file, open(text_path, 'w') as text_file:
            for caption in listdir(os.path.join(self.path, 'images')):
                if ' ' not in caption:
                    continue
                adj, noun = caption.split()
                if adj == 'adj':
                    continue

                for filepath in listdir(os.path.join(self.path, 'images', caption)):
                    assert (filepath.endswith('jpg'))
                    image_file.write(os.path.join(self.path, 'images', caption, filepath) + '\n')
                    text_file.write(caption + '\n')

    @staticmethod
    def generate_query_pair():
        query_nouns = [
            u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
            u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
            u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
            u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
            u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
            u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
            u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
            u'wheel', u'window', u'wool'
        ]
        captions = []
        noun_to_adjs = {}
        caption_to_image_id = {}
        num = 0
        with open(os.path.join(meta_path, '1.txt'), 'r') as file:
            for i, caption in enumerate(file):
                caption = caption.strip()
                captions.append(caption)
                adj, noun = caption.strip().split(' ')
                if caption not in caption_to_image_id.keys():
                    caption_to_image_id[caption] = []
                caption_to_image_id[caption].append(i)
                if noun not in noun_to_adjs:
                    noun_to_adjs[noun] = []
                if adj not in noun_to_adjs[noun]:
                    noun_to_adjs[noun].append(adj)
                num += 1
        pairs = []

        for i in range(100):
            noun = random.choice(query_nouns)
            while True:
                adj1 = random.choice(noun_to_adjs[noun])
                adj2 = random.choice(noun_to_adjs[noun])
                if adj1 == adj2:
                    continue
                first = random.choice(caption_to_image_id[f'{adj1} {noun}'])
                second = random.choice(caption_to_image_id[f'{adj2} {noun}'])
                pairs.append([first, second])
                break
        pairs.sort(key=lambda x: (x[0], x[1]))
        with open(os.path.join(query_path, 'MitStates.txt'), 'w') as file:
            for first, second in pairs:
                file.write(f'{first},{second}\n')

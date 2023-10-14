import os
import cv2
import glob
import spacy
import torch
import random
import spacy.cli
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.utils import shuffle
from PIL import Image


#spacy.cli.download("en_core_web_md")

spacy_de = spacy.load('de_core_news_lg')

print("Build start")

path = 'C:/Users/PC/2021-MLVU/SLT_project/frames/'

class Vocab_tokenizer():
    def __init__(self, freq_th, max_len):
        self.itos = {0 : "<blank>", 1 : "<PAD>", 2 : "<SOS>", 3 : "<EOS>", 4 : "<UNK>"}
        self.stoi = {"<blank>" : 0, "<PAD>" : 1, "<SOS>" : 2, "<EOS>" : 3, "<UNK>" : 4}
        self.freq_th = freq_th
        self.max_len = max_len

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        '''
        :param text: input translation string
        :return:     separated, lower-cased string
        '''
        # ["i", "love", "you"] * BS
        return[tok.text.lower() for tok in spacy_de.tokenizer(text)]

    def build_vocab(self, sentence_list):
        frequencies = {}
        idx = 5      # Since we pre-occupied three idx

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies :
                    frequencies[word] = 1
                else :
                    frequencies[word] += 1

                # Store the words over the frequency to self.itos and self.stoi
                if frequencies[word] == self.freq_th:
                    self.stoi[word] = idx
                    self.itos[idx]  = word
                    idx += 1

    # Integer tokenizer
    def numericalize(self, text):
        seqs = []
        for src in text:
            seq = [self.stoi["<PAD>"]]*self.max_len
            tokenized_src = self.tokenizer_eng(src)
            seq[0] = self.stoi["<SOS>"]                         # Patch SOS token at the front
            for idx, word in enumerate(tokenized_src):
                if idx == self.max_len:
                    break
                try:
                    seq[idx+1] = self.stoi[word]
                except :
                    seq[idx+1] = self.stoi["<UNK>"]
            seq[len(tokenized_src)+1]= self.stoi["<EOS>"]       # Patch EOS token at the end
            seqs.append(seq)
        return np.array(seqs)

    def stringnize(self, tokens):
        string = []
        for token in tokens:
            if token == 4 or token == 3 or token == 2 or token == 1 or token == 0:
                continue
            else:
                string.append(self.itos[token])
        return string




class SLT_dataset(Dataset):
    def __init__(self, root_dir, labels, glosses, targets, transform = None):
        self.root_dir = root_dir # base_path = 'C:/Users/PC/2021-MLVU/SLT_project/'
        self.labels = labels
        self.glosses = glosses
        self.translations = targets
        self.vocab = Vocab_tokenizer(freq_th = 1, max_len = 128)
        self.transform = transform

    def __len__(self):
        return len(self.translations)

    def __getitem__(self, index):
        # Load {features, translations}
        TensorPath = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/frames/' + self.labels[index] + '.pt'
        frames = None
        if os.path.exists(TensorPath):
            frames = torch.load(TensorPath)
            T, C, H, W = frames.shape
            if T != 0:
                max_len = 224
                # To mimic RandomCrop
                n_ = random.randint(-3, 3)
                if T <= max_len:
                    frames = frames[:, :, 14 + n_:-14 + n_, 5 + n_:-5 + n_]
                else:
                    indices = []
                    increment = T / max_len
                    indic = 0
                    for i in range(max_len):
                        indices.append(round(indic))
                        indic += increment
                    frames = frames[indices]
                    frames = frames[:, :, 14 + n_:-14 + n_, 5 + n_:-5 + n_]
            frames = frames / 255
        #print(self.labels[index], frames.shape, frames.dtype)
        gloss = torch.tensor(self.glosses[index])
        translation = torch.tensor(self.translations[index])
        if self.transform:
            frames = self.transform(frames)

        return frames, gloss, translation


class Padder:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        '''
        :param batch: batch images and texts
        :return:      padded batch images and padded texts
        '''

        # item : from __getitem__ : a single*BS (img, caption)
        frames = [item[0] for item in batch]
        frames = pad_sequence(frames, batch_first=True, padding_value=1.e-5)

        glosses = [item[1] for item in batch]
        glosses = pad_sequence(glosses, batch_first=True, padding_value=self.pad_idx)

        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return frames, glosses, targets

class RandomZeroOut(object):
    def __init__(self, p = 0.15):
        self.p = p

    def __call__(self, clip):
        n_frames = clip.size(0)
        n = int(n_frames * self.p)
        sampled_idx = random.sample(range(0, n_frames), n)
        clip[sampled_idx] = 1.e-5

        return clip



def get_loader(
        root_folder,
        labels, glosses, targets,
        BS=4,
        n_workers=0,
        transform = None,
        shuffle=True,
        pin_memory=True,
        random_seed=42
):
    dataset = SLT_dataset(root_folder, labels, glosses, targets, transform)

    dataset_len = len(dataset)
    data_inedx = list(range(dataset_len))
    # split_index = int(np.floor(dataset_len*test_split))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(data_inedx)
    train_indices = data_inedx

    # Initialize first token to "PAD"
    pad_idx = dataset.vocab.stoi["<PAD>"]
    # Define data sampler
    train_sampler = SubsetRandomSampler(train_indices)
    # Define dataloader. Be careful that sampler option is mutually exclusive with shuffle.
    loader = DataLoader(dataset=dataset, batch_size=BS,
                        num_workers=n_workers, sampler=train_sampler,
                        pin_memory=pin_memory, collate_fn=Padder(pad_idx=pad_idx))

    return loader, dataset, pad_idx




def main():
    data_path = 'C:/Users/Siryu_sci/2021-MLVU/SLT_project/'
    train_data = pd.read_csv(data_path + "PHOENIX-2014-T.train.corpus.csv", delimiter='|')
    val_data = pd.read_csv(data_path + "PHOENIX-2014-T.dev.corpus.csv", delimiter='|')
    test_data = pd.read_csv(data_path + "PHOENIX-2014-T.test.corpus.csv", delimiter='|')

    Traindata = pd.concat([train_data, val_data])
    max_len = 55

    # Define the tokenizer. data : translation, orth : gloss
    data_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len)
    orth_tokenizer = Vocab_tokenizer(freq_th=1, max_len=max_len)

    data_tokenizer.build_vocab(Traindata.translation)
    orth_tokenizer.build_vocab(Traindata.orth)
    # print(orth_tokenizer.stoi)

    targets = data_tokenizer.numericalize(Traindata.translation)
    glosses = orth_tokenizer.numericalize(Traindata.orth)
    labels = Traindata.name.to_numpy()

    print("Translation : ", targets.shape, len(data_tokenizer),
          "\n", "Glosses ; ", glosses.shape, len(orth_tokenizer))  # (7615, 300) 2948

    labels, targets, glosses = shuffle(labels, targets, glosses, random_state=42)
    # Train and validation (feature labels, translations)
    train_labels, train_glosses, train_translations = labels[:7115], glosses[:7115], targets[:7115]
    val_labels, val_glosses, val_translations = labels[7115:], glosses[7115:], targets[7115:]
    # test ''
    test_labels = test_data.name.to_numpy()
    test_glosses = orth_tokenizer.numericalize(test_data.orth)
    print(glosses.shape, test_glosses.shape)
    test_translations = data_tokenizer.numericalize(test_data.translation)

    transforms_ = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2, saturation=0.2),
        RandomZeroOut(p=0.1)]
    )

    print(data_tokenizer.itos[1], len(data_tokenizer.itos))

    train_loader, dataset, pad_idx = get_loader(data_path, train_labels,
                                train_glosses, train_translations, n_workers=4, transform=transforms_)

    for idx, (frames, glosses, captions) in enumerate(train_loader):
        print(frames.shape, frames.dtype, torch.mean(frames), torch.max(frames), torch.min(frames))
        plt.imshow(frames[3, 10, :, :, :].permute(1, 2, 0).numpy())
        plt.show()
        print(frames[3, :, 2, 100, 100])
        #print(frames[0, :, 0,10,10])
        print(captions.shape)

if __name__  == "__main__":
    main()
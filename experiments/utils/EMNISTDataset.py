import os
import pickle
import json

import torchvision.io
from torchvision.io import read_image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class EMNISTClassifierDataset(Dataset):
    def __init__(self, num_samples, labels_file, images_dir, fname_prefix='testing_image_'):
        """
        For a given set of propositions, we get their truth values given a set of inputs (images from an environment).
        """

        self.num_samples = num_samples
        self.images_dir = images_dir
        self.fname_prefix = fname_prefix

        with open(labels_file, 'rb') as f:
            self.labels = pickle.load(f)

    def __len__(self):
        return self.num_samples
        

    def __getitem__(self, item):
        img_fname = self.images_dir + self.fname_prefix + str(item) + '.png'
        image = read_image(img_fname, torchvision.io.ImageReadMode.GRAY)
        image = image.type(torch.float)
        label = torch.from_numpy(self.labels[item]).type(torch.float)
        return image, label

def main():
    dataset = EMNISTClassifierDataset(num_samples=100,
                                      labels_file='../../extended-mnist/output/mini_dataset/test_labels.pkl',
                                      images_dir='../../extended-mnist/output/mini_dataset/test/')
    print(len(dataset))
    image, label = dataset[0]
    image = image.permute(1, 2, 0)
    image = image / 255.0
    plt.imshow(image)
    plt.show()

    print(image.shape)

if __name__ == "__main__":
    main()

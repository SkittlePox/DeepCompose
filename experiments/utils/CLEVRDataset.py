# TODO Make a dataloader for CLEVR. Start with Propositional semantics.

import os
import pickle
import json

import torchvision.io
from torchvision.io import read_image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt


class CLEVR96ClassifierDataset(Dataset):
    def __init__(self, scene_file, images_dir, label_file, resize_dims=(80, 120)):
        """
        For a given set of propositions, we get their truth values given a set of inputs (images from an environment).
        """
        self.scene_file = scene_file
        self.images_dir = images_dir
        self.label_file = label_file
        self.resize = transforms.Resize(resize_dims)

        with open(self.scene_file, 'r') as f:
            self.scenes = json.load(f)['scenes']
        
        with open(self.label_file, 'r') as f:
            self.label_dict = json.load(f)

    def __len__(self):
        return len(self.scenes)
        

    def __getitem__(self, item):
        # Extract the image from self.images_dir, turn into tensor, and extract labels from self.label_file.

        img_fname = self.scenes[item]['image_filename']
        image = read_image(self.images_dir + img_fname, torchvision.io.ImageReadMode.RGB)
        image = self.resize(image.type(torch.float))
        label = (torch.tensor(self.label_dict[img_fname]['96count'], dtype=torch.float) > 0).type(torch.float)
        return image, label


def main():
    # file = open('../images/random_states.pkl', 'rb')
    dataset = CLEVR96ClassifierDataset(scene_file="../../clevr-refplus-dcplus-dataset-gen/output/scenes/clevr_ref+_cogent_valA_scenes.json",
                                       images_dir="../../clevr-refplus-dcplus-dataset-gen/output/images/valA/",
                                       label_file="../../clevr-refplus-dcplus-dataset-gen/output/labels/clevr_ref+_cogent_valA_labels.json")

    print(len(dataset))
    image, label = dataset[0]
    image = image.permute(1, 2, 0)
    image = image / 255.0
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()

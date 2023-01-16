import os
import pickle
from skimage import io
from torch.utils.data import Dataset, DataLoader


class PropositionDataset(Dataset):
    def __init__(self, root_dir, label_dict):
        """
        For a given set of propositions, we get their truth values given a set of inputs (images from an environment).
        """
        self.root_dir = root_dir
        self.label_dict = label_dict

    def __len__(self):
        return len(self.label_dict.items())

    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir,
                                f"state-{item}.png")
        image = io.imread(img_name)
        label = self.label_dict[f"state-{item}"]

        return image, label


def main():
    file = open('../images/random_states.pkl', 'rb')
    dataset = PropositionDataset(root_dir="../images/random_states/", label_dict=pickle.load(file))

    print(dataset.__getitem__(1))


if __name__ == "__main__":
    main()

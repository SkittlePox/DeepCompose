from torch.utils.data import Dataset, DataLoader, default_collate


class PropositionDataset(Dataset):
    def __init__(self, proposition_list, inputs, values):
        """
        For a given set of propositions, we get their truth values given a set of inputs (images from an environment).
        """
        self.proposition_list = proposition_list
        self.inputs = inputs
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.inputs[item], self.values[item]


def main():
    pass


if __name__ == "__main__":
    main()

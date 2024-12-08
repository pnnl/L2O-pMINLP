from sklearn.model_selection import train_test_split
import torch

from neuromancer.dataset import DictDataset


def getDatasetQradratic(num_data, num_vars, test_size=0, val_size=0, random_state=42):
    # dictionary of parameters
    paramdict = genParamQradratic(num_data, num_vars)
    # dictionary datasets
    datasets = getDataset(paramdict, test_size=test_size, val_size=val_size, random_state=random_state)
    return datasets


def genParamQradratic(num_data, num_vars):
    # data sample from uniform distribution
    p_low, p_high = 1.0, 11.0
    p_samples = torch.FloatTensor(num_data, num_vars).uniform_(p_low, p_high)
    return {"p":p_samples}


def getDatasetRosenbrock(num_data, num_vars, test_size=0, val_size=0, random_state=42):
    # dictionary of parameters
    paramdict = genParamRosenbrock(num_data, num_vars)
    # dictionary datasets
    datasets = getDataset(paramdict, test_size=test_size, val_size=val_size, random_state=random_state)
    return datasets


def genParamRosenbrock(num_data, num_vars):
    # data sample from uniform distribution
    p_low, p_high = 0.5, 6.0
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    a_low, a_high = 0.2, 1.2
    a_samples = torch.FloatTensor(num_data, num_vars - 1).uniform_(p_low, p_high)
    return {"p":p_samples, "a":a_samples}


def getDatasetRatrigin(num_data, num_vars, test_size=0, val_size=0, random_state=42):
    # dictionary of parameters
    paramdict = genParamRatrigin(num_data, num_vars)
    # dictionary datasets
    datasets = getDataset(paramdict, test_size=test_size, val_size=val_size, random_state=random_state)
    return datasets


def genParamRatrigin(num_data, num_vars):
    # data sample from uniform distribution
    p_low, p_high = 2, 6
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    a_low, a_high = 6, 15
    a_samples = torch.FloatTensor(num_data, num_vars).uniform_(p_low, p_high)
    return {"p": p_samples, "a": a_samples}


def getDataset(paramdict, test_size=0, val_size=0, random_state=42):
    datasets = {"train":None}
    ind_train = range(len(next(iter(paramdict.values()))))
    if test_size: # train & test
        # data split
        ind_train, ind_test = train_test_split(ind_train,
                                               test_size=test_size,
                                               random_state=random_state,
                                               shuffle=True)
        # dictionary dataset
        datasets["test"] = DictDataset({key: param[ind_test] for key, param in paramdict.items()}, name="test")
    if val_size:
        # data split
        ind_train, ind_dev = train_test_split(ind_train,
                                              test_size=val_size,
                                              random_state=random_state,
                                              shuffle=True)
        # dictionary dataset
        datasets["dev"] = DictDataset({key: param[ind_dev] for key, param in paramdict.items()}, name="dev")
    # dictionary dataset
    datasets["train"] = DictDataset({key: param[ind_train] for key, param in paramdict.items()}, name="train")
    return tuple(datasets.values())


if __name__ == "__main__":
    # datasets
    datasets = getDatasetRosenbrock(num_data=100, num_vars=5, test_size=20, val_size=10)
    for dataset in datasets:
        print(len(dataset))
        print(dataset[0])
import random
#import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import numpy as np

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


def getDatasetMarkowitz(num_data, num_vars, test_size=0, val_size=0, random_state=42):
    # dictionary of parameters
    exp_returns, cov_matrix, paramdict = genParamMarkowitz(num_data, num_vars, random_state)
    # dictionary datasets
    datasets = getDataset(paramdict, test_size=test_size, val_size=val_size, random_state=random_state)
    return exp_returns, cov_matrix, datasets

def genParamMarkowitz(num_data, num_vars, random_state=42):
    # descriptive data on S&P 500
    #sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    #sp500_df = pd.read_html(sp500_url)[0]
    # ticker symbol
    #sp500_symbols = sp500_df['Symbol'].tolist()
    # random selection
    #local_random = random.Random(random_state)
    #selected_stocks = local_random.sample(sp500_symbols, num_vars)
    # daily closing prices
    #data = yf.download(selected_stocks, start="2021-01-01", end="2022-01-01")["Close"]
    # daily returns
    #returns = data.pct_change().dropna()
    # expected returns
    #exp_returns = returns.mean().values
    # covariance matrix
    #cov_matrix = returns.cov().values
    # expected returns
    exp_returns = np.random.uniform(0.002, 0.01, num_vars)
    # covariance matrix
    A = np.random.rand(num_vars,num_vars)
    # positive semi-definite matrix
    cov_matrix = A @ A.T / 1000
    # parameters
    p_low, p_high = max(min(exp_returns),0), max(exp_returns)
    p_samples = torch.FloatTensor(num_data, 1).uniform_(p_low, p_high)
    return exp_returns, cov_matrix, {"p": p_samples}


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
    print("Rosenbrock")
    datasets = getDatasetRosenbrock(num_data=100, num_vars=5, test_size=20, val_size=10)
    for dataset in datasets:
        print(len(dataset))
        print(dataset[0])
    print()

    print("Markowitz")
    exp_returns, cov_matrix, datasets = getDatasetMarkowitz(num_data=100, num_vars=5, test_size=20, val_size=10)
    print("Expected Returns:")
    print(exp_returns)
    # covariance matrix
    print("Covariance Matrix:")
    print(cov_matrix)
    for dataset in datasets:
        print(len(dataset))
        print(dataset[0])
    print()

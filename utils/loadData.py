import torch
from scipy.io import loadmat
from sklearn import preprocessing


def loadData(args):
    dataset = loadmat(args.path + '/' + args.DS + '.mat')
    if ('X' in dataset):
        x = torch.tensor(dataset['X'][:])
        y = torch.tensor(dataset['Y']).squeeze().numpy()
        if 'csc_matrix' in str(type(x)):
            x = torch.tensor(x.todense())
    elif ('feature' in dataset):
        x = torch.tensor(dataset['feature'][:])
        y = torch.tensor(dataset['label']).squeeze().numpy()
    else:
        x = torch.tensor(dataset['fea'][:])
        y = torch.tensor(dataset['gnd']).squeeze().numpy()
        if 'csc_matrix' in str(type(x)):
            x = torch.tensor(x.todense())
    minmax = preprocessing.MinMaxScaler()
    x = torch.tensor(minmax.fit_transform(x), dtype=torch.float)
    return x, y

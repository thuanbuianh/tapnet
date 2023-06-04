import numpy as np
import sklearn
import sklearn.metrics
import torch
from pyts.datasets import fetch_uea_dataset
from sklearn.preprocessing import LabelEncoder


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')

    return s == 'True'


def standardise(data, mean=0, std=1, isTest=False):
    if not isTest:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / std, mean, std

    return (data - mean) / std


def load_raw_ts(path, dataset, tensor_format=True):
    x_train, x_test, y_train, y_test = fetch_uea_dataset(dataset,
                                                         data_home=path,
                                                         return_X_y=True)
    x_train, mean, std = standardise(x_train)
    x_test = standardise(x_test, mean, std, isTest=True)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    ts = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)
    nclass = int(np.amax(labels)) + 1

    train_size = y_train.shape[0]

    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_val = range(train_size, total_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_val, idx_test, nclass, mean, std


def load_test_ts(path, dataset, data_home='./data/raw/', tensor_format=True):
    x_train, _, y_train, _ = fetch_uea_dataset(dataset,
                                               data_home=data_home,
                                               return_X_y=True)
    x_train, mean, std = standardise(x_train)
    x_test = np.loadtxt(path, delimiter=',')
    x_test = np.expand_dims(x_test, axis=0)
    x_test = standardise(x_test, mean, std, isTest=True)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = np.array([-1])

    ts = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    train_size = y_train.shape[0]
    total_size = labels.shape[0]
    idx_train = range(train_size)
    idx_test = range(train_size, total_size)

    if tensor_format:
        ts = torch.FloatTensor(np.array(ts))
        labels = torch.LongTensor(labels)

        idx_train = torch.LongTensor(idx_train)
        idx_test = torch.LongTensor(idx_test)

    return ts, labels, idx_train, idx_test, le


def accuracy(output, labels):
    preds = output.max(1)[1].cpu().numpy()
    labels = labels.cpu().numpy()
    accuracy_score = (sklearn.metrics.accuracy_score(labels, preds))

    return accuracy_score


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)


def output_conv_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1

    return output


def dump_embedding(proto_embed,
                   sample_embed,
                   labels,
                   dump_file='./plot/embeddings.txt'):
    proto_embed = proto_embed.cpu().detach().numpy()
    sample_embed = sample_embed.cpu().detach().numpy()
    embed = np.concatenate((proto_embed, sample_embed), axis=0)

    nclass = proto_embed.shape[0]
    labels = np.concatenate((np.asarray(
        [i for i in range(nclass)]), labels.squeeze().cpu().detach().numpy()),
                            axis=0)

    with open(dump_file, 'w') as f:
        for i in range(len(embed)):
            label = str(labels[i])
            line = label + "," + ",".join(
                ["%.4f" % j for j in embed[i].tolist()])
            f.write(line + '\n')

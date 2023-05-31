from __future__ import division
from __future__ import print_function
import argparse
from utils import *

parser = argparse.ArgumentParser()
# model saving setting
parser.add_argument('--save_path', type=str, default="./models",
                    help='the path of saving models.')
# dataset settings
parser.add_argument('--data_path', type=str, default="./data/raw",
                    help='the path of data.')
parser.add_argument('--dataset', type=str, default="NATOPS", #NATOPS
                    help='time series dataset. Options: See the datasets list')
args = parser.parse_args()

data_path = args.data_path
dataset = args.dataset
features, labels, idx_train, idx_test = load_test_ts(data_path, dataset)

model_path = args.save_path
model = torch.load(model_path)
model.eval()

input = (features, labels, idx_train, idx_test)
output, proto_dist = model(input)
test_results = output[idx_test]
print(test_results)


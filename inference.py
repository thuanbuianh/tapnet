import pandas as pd
import streamlit as st
import torch

from models import TapNet
from utils import load_test_ts

dataset_option = st.selectbox(
    'Please choose one of these dataset: ',
    ('ArticularyWordRecognition', 'BasicMotions', 'NATOPS'))
st.write(f'You have chosen {dataset_option} dataset')
file = st.file_uploader(
    f'Please upload {dataset_option} test file (in csv format)', type='csv')
if file is not None:
    model_settings = {
        'ArticularyWordRecognition': {
            'nfeat': 9,
            'len_ts': 144,
            'layers': [500, 300],
            'nclass': 25,
            'dropout': 0.5,
            'use_lstm': True,
            'use_cnn': True,
            'filters': [256, 256, 128],
            'dilation': 5,
            'kernels': [8, 5, 3],
            'use_metric': False,
            'use_rp': True,
            'rp_params': [3, 6],
            'lstm_dim': 128
        },
        'NATOPS': {
            'nfeat': 24,
            'len_ts': 51,
            'layers': [500, 300],
            'nclass': 6,
            'dropout': 0.5,
            'use_lstm': True,
            'use_cnn': True,
            'filters': [256, 256, 128],
            'dilation': 1,
            'kernels': [8, 5, 3],
            'use_metric': False,
            'use_rp': True,
            'rp_params': [3, 16],
            'lstm_dim': 128
        },
        'BasicMotions': {
            'nfeat': 6,
            'len_ts': 100,
            'layers': [500, 300],
            'nclass': 4,
            'dropout': 0.5,
            'use_lstm': True,
            'use_cnn': True,
            'filters': [256, 256, 128],
            'dilation': 1,
            'kernels': [8, 5, 3],
            'use_metric': False,
            'use_rp': True,
            'rp_params': [3, 4],
            'lstm_dim': 128
        }
    }

    setting = model_settings[dataset_option]

    model = TapNet(nfeat=setting['nfeat'],
                   len_ts=setting['len_ts'],
                   layers=setting['layers'],
                   nclass=setting['nclass'],
                   dropout=setting['dropout'],
                   use_lstm=setting['use_lstm'],
                   use_cnn=setting['use_cnn'],
                   filters=setting['filters'],
                   dilation=setting['dilation'],
                   kernels=setting['kernels'],
                   use_metric=setting['use_metric'],
                   use_rp=setting['use_rp'],
                   rp_params=setting['rp_params'],
                   lstm_dim=setting['lstm_dim'])

    model_path = './models'
    model.load_state_dict(
        torch.load(f'{model_path}/{dataset_option}.pth',
                   map_location=torch.device('cpu')))
    model.eval()

    features, labels, idx_train, idx_test, le = load_test_ts(
        file, dataset_option)
    input = (features, labels, idx_train)
    output, proto_dist = model(input)
    test_results = output[idx_test]
    test_ts = pd.DataFrame(features[idx_test].squeeze().T).astype(float)
    st.write('Visualising test time series...')
    st.line_chart(test_ts)
    pred_label = le.inverse_transform([torch.argmax(test_results).item()])[0]
    st.write(f'The predicted result is {pred_label}')

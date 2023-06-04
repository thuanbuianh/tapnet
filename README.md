# Enhancing Attentional Prototype Network with Squeeze-and-Excitation Blocks.

## Introduction
This is a Pytorch implementation of a multivariate time series classification method based on Attentional Prototype Network for the paper **TapNet: Multivariate Time Series Classification with Attentional Prototype Network** and Squeeze-and-Excitation blocks for the paper **Squeeze-and-Excitation Networks**.

## Technical Overview
There are 2 modifications between this version (SE-TapNet) and the original Attentional Prototype Network (TapNet):
- Stacking Squeeze-and-Excitation (SE) blocks after each convolutional layer to enhance the performance of the model.
- Standardising time series.
With these modifications, some datasets are used to evaluate the performances. The results are below:

|          Dataset          |   TapNet  |  SE-TapNet |
|:-------------------------:|:---------:|:----------:|
| ArticularyWordRecognition | **0.987** |    0.983   |
|     AtrialFibrillation    |   0.333   |    0.333   |
|        BasicMotions       |     1     |      1     |
|   HandMovementDirection   |   0.378   | **0.5946** |
|           NATOPS          |   0.939   | **0.9667** |
|          PEMS-SF          |   0.751   | **0.8382** |

A basic interface is also developed by using Streamlit.

## Installation
1. To train models, please follow instructions in this [notebook](https://colab.research.google.com/drive/1nB46gCefj7yhCyCRduVeFHaNfITGha2U?usp=sharing).
2. To install the application for testing, please build a image from the Docker file (it will take approximately 5~6 minutes)
```
docker build -t <your-image-name> .
```
## Usage
Please follow these steps in order to install and use Streamlit UI for testing:
1. Run the image built in previous step
```
docker run -p 8051:8051 <your-image-name>
```
2. Browse to [http://localhost:8501](http://localhost:8501) to access the Streamlit UI.
3. Choose a dataset and upload your test file. Note that the test file should contain only ONE multivariate time series and must be in csv format. You can use test files in test folder.

For demo purpose, only 3 datasets having high accuracy (**ArticularyWordRecognition**, **BasicMotions** and **NATOPS**) are available for testing. 
## Paper

```
@inproceedings{zhang2020tapnet,
  title={TapNet: Multivariate Time Series Classification with Attentional Prototypical Network.},
  author={Zhang, Xuchao and Gao, Yifeng and Lin, Jessica and Lu, Chang-Tien},
  booktitle={AAAI},
  pages={6845--6852},
  year={2020}
}
```
```
@inproceedings{hu2018squeeze,
  title={Squeeze-and-excitation networks},
  author={Hu, Jie and Shen, Li and Sun, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={7132--7141},
  year={2018}
}
```
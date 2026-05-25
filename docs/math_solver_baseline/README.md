# Quick Start Guide

This repository contains a Math Solver model that can be trained and tested using the provided scripts. Follow the steps below to set up the environment, train the model, and test it on a designated test set.

## Setup

### Clone the Repository

To get started, clone the repository using the following command:

```bash
git clone https://github.com/AI-FDU/Math_Solver.git
```

### Prepare the Environment

If you are using the modelscope (魔搭), you do not need to install any additional environment. Otherwise, you can set up the required environment by running:

```bash
pip install transformers modelscope peft swanlab
```

## Training

To initiate the training process, execute the following command:

```bash
python qwen_ft.py
```

## Testing

Once the training is complete, you can test the model on a designated test set by running:

```bash
python infer.py
```

This will generate a `submit.csv` file, which can be directly submitted to the competition platform at [DataFountain](https://www.datafountain.cn/competitions/467/submits).

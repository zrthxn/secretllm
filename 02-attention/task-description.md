<!-- <img width="100" src="https://tu-dresden.de/++theme++tud.theme.webcms2/img/tud-logo.svg">
<img width="100" src="https://scads.ai/wp-content/themes/scads2023/assets/images/logo.png"> -->
<!-- <h3>Behind the Secrets of Large Language Models</h3>
<p>Exercise 2</p> -->

### Exercise 2
# Attention for time-series Prediction

This time we will learn how to implement attention and how it can improve prediction of time-series data.

- [Attention for time-series Prediction](#attention-for-time-series-prediction)
  - [HPC Quickstart](#hpc-quickstart)
    - [Task 0: Run a Notebook on HPC](#task-0-run-a-notebook-on-hpc)
  - [PANJAPAN Dataset](#panjapan-dataset)
    - [Task 1: Read and Prepare the Dataset](#task-1-read-and-prepare-the-dataset)
  - [Build and Train an RNN](#build-and-train-an-rnn)
    - [Task 2: Write a simple LSTM model](#task-2-write-a-simple-lstm-model)
    - [Task 3: Train your LSTM model](#task-3-train-your-lstm-model)
    - [Task 4: Plot results on test split](#task-4-plot-results-on-test-split)

## HPC Quickstart

We assume you have signed up on the HPC with the link/QR-code provided.
Read the HPC Quickstart guide that we have prepared and ensure that you are able to login to jupyterhub.hpc.tu-dresden.de 
with your ZIH login credentials, as well as login via `ssh`.

```bash
ssh login2.alpha.hpc.tu-dresden.de -l <ZIH-Login>
```

Your personal data like code, notebooks, plots, datasets you downloaded etc. should be kept in your home directory `/home/<ZIH-Login>`.
The project directory is available for project data that you want to share with other project members `/projects/p_scads_llm_secrets`.

> **Note: Login to the HPC is only possible on the university netowork or via the VPN!**

### Task 0: Run a Notebook on HPC
Read this using pandas and pre-process it to be ready for training.

## PANJAPAN Dataset
We will use a time-series dataset called PANJAPAN.
You can download the dataset from Huggingface: LINK.
The dataset is already available on the HPC under `/projects/p_scads_llm_secrets/datasets/PANJAPAN`

The dataset contains many samples of time-series like these in CSV files. Each file has a value column
and a timestamp column. Many of the timeseries here have seasonalities which we should
be able to predict with our neural netowork.

Here is what the dataset looks like. 
<img src="./img/panjapan-sample.png">

### Task 1: Read and Prepare the Dataset
Read this using pandas and pre-process it to be ready for training.

```py
def read_data():
  ...
```

## Build and Train an RNN

Sequence-to-Sequence or Seq2Seq models are a machine learning architecture designed for tasks 
involving sequential data, like time-series forecasting or language translation. 
**We take an input sequence, processes it, and generate an output sequence.**

Recurrent Neural Networks are a common architecture to transform sequences. Vanilla RNNs suffer from the problem of 
vanishing gradient, where the gradient of loss becomes too small over many time steps to be useful for training. 
Thus these are rarely used, in favour of LSTM or GRU which are more common.

### Task 2: Write a simple LSTM model
Using PyTorch, write a very simple LSTM model that takes in a sequence of values and returns another sequence.
Use the reference of...

```py
from torch import nn

# Here we will define our neural network as a class
class LSTM(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    
    # TODO: Use the LSTM layer and a linear layer from torch.nn
  
  def forward(self, seq):
    # TODO: Write a forward pass through the 
    #       LSTM layer and then a linear layer

    return prediction

rnn = LSTM(input_size = ..., output_size = ...)
```

### Task 3: Train your LSTM model
We will now and a training loop for the dataset

```py
from torch.optim import Adam

optimizer = Adam(params=rnn.parameters(), lr=5e-3)

for epoch in range(EPOCHS):
  for i, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch}/{EPOCHS}"):
    optimizer.zero_grad()
    # TODO: Complete the training loop
    ...
```

### Task 4: Plot results on test split




<!-- links -->
[torch.nn.lstm][https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html]
[tutorial-rnns][https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html]
[recurrent-layers][https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html#recurrent-layers]
[lstm-tutorial][https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/]
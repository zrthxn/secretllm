### Exercise 4
# Training a Language Model

This time you will actually build a relatively small language model using a dataset of your choosing
and we will show you a "large" language model. We will also look at various LLM architectures.

## Finding a Dataset

The first task in building any machine learning application is to find a dataset to work with. There 
are many online repositories of datasets that you can use for this.

The most popular website for NLP related datasets (and models) is [HuggingFace](huggingface.co).
For this task we recommend that you try to find a dataset in your language or a language you understand.

1. Go to https://huggingface.co/datasets
2. Under **Main** first filter by *Modalities*, select **Text**. 
   Under **Tasks**, select **Text Generation**. 
   Under **Libraries**, select **Datasets**.
3. Under **Languages** select your language. This should now give you a list of suitable datasets that we can use.
4. Choose any of the datasets you see which is not too big in size and check it's description
   for how to load the dataset. This may be different for different datasets. 

<div style="display: flex;">
    <img width="55%" src="images/dataset-type.png">
    <img width="45%" src="images/dataset-languages.png">
</div>

Of course you can choose a dataset according to what you want to build.
Other interesting ideas for datasets may be code datasets, math datasets, song lyrics datasets.
Finding and getting them ready for use may be more work but would be a good exercise.
Remember that a machine learning model is only as good as the dataset it uses.

## Loading and Running

The example in the accompanying notebook will show a Wikipedia dataset in Belarusian.

## Training

## Evaluation

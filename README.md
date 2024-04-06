# Out-of-Distrbution Detection in a VQ-VAE for Enhanced Arc Welding Predictions

## Environment
We recommand using the [devcontainer](.devcontainer) to run the code.

Otherwise, the packages can be installed using conda and the environment [file](.devcontainer/environment.yml) with the following command:
```bash
conda env create -n vqvae-transformer python=3.11 -f .devcontainer/environment.yml
conda activate vqvae-transformer

``` 


## Dataset
The dataset is available at [zenodo](https://zenodo.org/records/10017718). 
Please download the processed dataset and put it in the `data` folder.



## OOD Detection Models
For the investigation of OOD data on the VQ loss there are scripts prepared for systematically training models while deliberately omitting data from the training set.
```bash
python train_vqvaes_for_OOD.py 
```
### Arguments

|Argument|Description||
|-|-|-|
|***Type Selection***|*(select at least one)*||
|**--ex**| train VQ-VAE models on overlap joints and T-joints|
|**--no-ex**| *dont* train models on overlap joints and T-joints |(default)
|**--param**| train VQ-VAE models with data split by parameter ranges|
|**--no-param**| *dont* train models with data split by parameter ranges |(default)
||||
|***Split Selection***|||
|**--param-splits**| Types of parameter splits for training |default=['vs', 'vd', 'vs-inv', 'vd-inv'] 
|    |vs = low, vs-inv = high |voltage settings
|    |vd = low, vd-inv = high |wire-feed speed
|**--ex-splits**| Types of splits on experiments |default=["ex", "ex-inv"]
|    |ex = train on experiments 1 and 2 | overlap joint
|     |ex-inv = train on experiment 3| T-joint 
||||
|***Hyperparameters***|||
|**--epochs**| list of training epochs| default=[10, 20, 30, 50]
|**--num_embeddings**| Number of embedding values| default=[16, 64, 256]
|**--betas**| Beta values |default=[1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 1]

### Examples
##### Example 1
```bash
python train_vqvaes_for_OOD.py --ex --no-param --ex-splits ex --epochs 10 50 --num_embeddings 256 --betas 0.25 1e-06
```
Train VQ-VAE models on overlap joint runs with 10 and 50 epochs, 256 embeddings, and with 0.25 and 1e-06 as beta values
##### Example 2
```bash
python train_vqvaes_for_OOD.py --ex --param --param-splits vs vs-inv vd vd-inv --ex-splits ex ex-inv --epochs 50 --num_embeddings 16 64 --betas 0.25
```
Train VQ-VAE models on low and high voltage settings and wire-feed speed for 50 epochs with beta=0.25 and
with 16 and 64 embeddings.

## Plotting VQ Loss
[VQ_Loss_Plotter](VQ_Loss_Plotter.py) was made to calculate and plot losses for the previously trained models. In the [Notebook](Plotting.ipynb) are some examples on how to use it. The Notebook also contrains some extra utility, including plotting raw data cycles.
### Boxplots
```python
VQ_Loss_Plotter.boxplots(
    data_splits: list,
    epochs: list,                       
    embeddings: list,
    betas: list,
    figsize=(10, 10),
    image_path: str = "images/",
    title=None,
    file_name=None,
    rotation=45,
    auto_filenames=False,
    xlim=None,
    ylim=None,
    custom_selection=None, 
    amount=0
)
```
Makes boxplots for the loss of full runs.
### Loss Time Series
```python
VQ_Loss_Plotter.thresholds(
    data_splits: list,
    epochs: list,
    embeddings: list,
    betas: list,
    figsize=(10, 10),
    title: str = None,
    image_path: str = "images/",
    file_name=None,
    amount=0,
    xlim=None,
    ylim=None
)
```
Makes times series plots over batch losses of welding runs with mean, Q1, and Q3 as orientation and possible thresholds for OOD detection.
### Noise
```python
VQ_Loss_Plotter.test_on_noise(
    data_splits,
    epochs,
    embeddings,
    betas,
    image_path: str = "images/Random/",
    figsize=(10, 10)
)
```
Makes boxplots to compare VQ losses of the model(s) when specific data splits
are used as input versus a run with random generated noise.
### Trained on Full Dataset
```python
VQ_Loss_Plotter.make_boxplot_all_data(
    title,
    data_split,
    image_path,
    file_name,
    model_path,
    figsize=(10, 10),
    rotation=45,
    custom_selection=None,
    amount=0,
    xlim=None,
    ylim=None
)
```
Makes a boxplot comparison of the loss of the welding runs of a data split when the model specified was trained on the full data set.
### Plotting Raw Data
Download **raw_data.csv** from [zenodo](https://zenodo.org/records/10017718) first. Then use
```bash
python raw_data_to_pickle.py
```
to setup and save the dataloaders for the raw data for each welding experiment.
The dataloaders can then be used as exemplified in the [Notebook](Plotting.ipynb).

## Quality Prediction
#### Model
- [VQ-VAE-Transformer](model/transformer_decoder.py)
- [VQ-VAE-Patch](model/vq_vae_patch_embedd.py)


#### Training
To train the model first the VQ-VAE must be trained. 
```bash
python train_reconstruction_embedding.py
```

Then the VQ-VAE-Transformer can be trained.
```bash
python train_transformer_mtasks.py.py --vq-model="path to trained vq model"
```
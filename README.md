<h1 align="center"> Exploring the Efficacy of Egocentric Observations in Reinforcement Learning Across Various Tasks </h1>

# Necessary library installations
```bash
conda create -n comparative-study -c conda-forge python=3.7
```
```bash
conda activate comparative-study
```
```bash
pip install setuptools==65.5.0 pip==21
```
```bash
pip install wheel==0.38.0
```
```bash
pip install -r requirements.txt
```
```bash
pip install wandb
```
# Order which code has to be executed
```bash
python train.py                   # to train your models
```
```bash
python visualise.py              # to visualise the learnt policies
```
```bash
python evaluate.py               # to evaluate the learnt policies
```
```bash
python training_plots.py               # to evaluate the learnt policies
```
etc

Note modifications to the code would be needed depending on how you run the experiment and the folders at which you going to store the CSV files that record the training data

Pretrained models have been provided to get you started with visualising some learnt policies. You might need to comment some lines out and make modifications to some variables to get started.

Note: github is refusing to accept uploads for the navigation sample models, so comment Navi lines out in visualise.py (it's only 2 lines). Or better yet just train yourself Navi agents to visualise, should take a while though, evaluate.py needs you to do it

Enjoy!

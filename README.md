This repo is for Fish classification.

To start:
- Have your data with metadata.csv in a directory.
- Create a config file using HGNN/train/ConfigParserWriter-*
- train a model using train.py. e.g.: python3 train.py --cuda=7 --name="learningRateTest" --experiments="/home/elhamod/HGNN/experiments/" --data="/data/BGNN_data"
- analyze the data using jupyter notebooks under HGNN/analyse/ .e.g Analyze trial, Analyze experiments.
- Once trained and analyzed, an experiment folder with name <experiments>/<name> will have been created, with a "models" folder that has all the trained model, a "results" folder with the analysis, and a "datasplits" folder with the indexes of images used for train/val/test.

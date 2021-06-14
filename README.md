This repo is for Fish classification.

To train:
- Have your data with metadata.csv in a directory. A sample file can be found in HGNN_repo_artifacts.zip. Your image data needs to be in the torchvision ImageFolder format, with train/val/test subfolders.
- Create a config file using HGNN/train/ConfigParserWriter-SelectExperiments.ipynb
- train a model using HGNN/train/train.ipynb.

  
To view results:
- analyze the data using jupyter notebooks HGNN/analyse/Analyze experiments.ipynb.
- Once trained and analyzed, an experiment folder with name <experiments>/<name> will have been created, with a "models" folder that has all the trained model, a "results" folder with the analysis.
- You can also look at HGNN/analyse/Analyze multiple experiments-*.ipynb for seeing the results in our MEE paper
- To see Saliency map results, you can look at HGNN/analyse/Saliency Maps - *.ipynb files. 

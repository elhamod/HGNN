from setuptools import setup, find_packages

setup(name='HGNN', version='1.0', packages=find_packages(exclude=['HGNN.old', ])) #, include=["CNN.py","configParser.py", "CSV_processor.py", "dataLoader.py" ] 
# setup(name='HGNNhelpers', version='1.0',  package_dir={"": "helpers"}, packages=find_packages(where="helpers" )) # , include=["color.PCA.py","config_plots.py", "ZCA.py", "TrialStatistics.py", "confusion_matrix_plotter.py", "earlystopping.py", "dataset_normalization.py" ] 
# ,  package_dir={"": "train"}

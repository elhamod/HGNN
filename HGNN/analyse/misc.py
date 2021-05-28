import torch
import pandas as pd
from tqdm import tqdm
import os

# Given a dataset and model's predictions, it returns two dataframes of well classified and misclassified examples.
def get_classification_df(dataset, predProblist_, lbllist, modelName, experimentPathAndName):

    df_misclassified = pd.DataFrame(columns=['file name', 'true label', 'probability of true label', 'predicted label'])
    df_correctlyclassified = pd.DataFrame(columns=['file name', 'true label', 'probability of true label', 'predicted label'])

    # get probability of correct prediction and true label
    
    _, predlist = torch.max(predProblist_, 1)
    lbllist = lbllist.reshape(lbllist.shape[0], -1)
    predProblist = predProblist_.gather(1, lbllist)
    predProblist = predProblist.reshape(1, -1)
    predProblist = predProblist[0]

    # sort through
    predProblist, indices = torch.sort(predProblist)
    predlist = predlist[indices]
    lbllist = lbllist[indices]

    for i, lbl in tqdm(enumerate(lbllist), total=len(lbllist)):
        prd = predlist[i]
        prdProb = predProblist[i]

        if torch.cuda.is_available():
            lbl = lbl.cpu()
            prd = prd.cpu()
            prdProb = prdProb.cpu()

        s = dataset[indices[i]]
        row = {'file name' : s['fileName'] , 
               'true label' : int(lbl.numpy()), 
               'probability of true label': float(prdProb.numpy()),
               'predicted label' : int(prd.numpy()),
              'original_index': int(indices[i])}

        if(lbl != prd):
            df_misclassified = df_misclassified.append(row, ignore_index=True)
        else:
            df_correctlyclassified = df_correctlyclassified.append(row, ignore_index=True)


    df_misclassified = df_misclassified.sort_values(by=[ 'true label', 'probability of true label'])
    df_correctlyclassified = df_correctlyclassified.sort_values(by=['true label', 'probability of true label'])
    
    df_misclassified.to_csv(os.path.join(experimentPathAndName, modelName, 'misclassified examples.csv'))
    df_correctlyclassified.to_csv(os.path.join(experimentPathAndName, modelName, 'correctly classified examples.csv'))
    
    return df_misclassified, df_correctlyclassified
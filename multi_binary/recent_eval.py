import os
from utils import *
import pandas as pd

TEST_LABEL_TO_ID = {'no_relation':0, 'org:top_members/employees':1, 'org:members':2, 'org:product':3, 
'per:title':4, 'org:alternate_names':5, 'per:employee_of':6, 'org:place_of_headquarters':7, 
'per:product':8, 'org:number_of_employees/members':9, 'per:children':10, 'per:place_of_residence':11, 
'per:alternate_names':12, 'per:other_family':13, 'per:colleagues':14, 'per:origin':15, 
'per:siblings':16, 'per:spouse':17, 'org:founded':18, 'org:political/religious_affiliation':19, 
'org:member_of':20, 'per:parents':21, 'org:dissolved':22, 'per:schools_attended':23, 
'per:date_of_death':24, 'per:date_of_birth':25, 'per:place_of_birth':26, 'per:place_of_death':27, 
'org:founded_by':28, 'per:religion':29}

binary_path = '/opt/ml/code/prediction/recent/depot-recent-binary.csv'
binary_predictions = pd.read_csv(binary_path)

# use predicted relation overwrite the original relation

for type_id in range(0, 4):
    multi_path = '/opt/ml/code/prediction/recent/depot-recent-%d.csv' % (type_id)
    preds = pd.read_csv(multi_path)
    label_list = list(LABEL_TO_ID[type_id].keys())
    label_idx = [TEST_LABEL_TO_ID[l] for l in label_list]
    # pred_probs 30개로 늘려주기
    pred_probs = []
    for idx, r in preds.iterrows():
        zeros = np.zeros(30)
        # id 찾아서 바꿔주기
        for jdx, p in zip(label_idx, eval(r['probs'])):
            zeros[jdx] = p
        if binary_predictions.loc[r['id'],'pred_label'] != 'no_relation':
            binary_predictions.loc[r['id'],['pred_label','probs']] = [r['pred_label'],str(zeros)]

binary_predictions.to_csv('/opt/ml/code/prediction/recent/depot-recent-final.csv')

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

for type_id in range(0, 12):
    multi_path = '/opt/ml/code/prediction/recent/depot-recent-%d.csv' % (type_id)
    preds = pd.read_csv(multi_path)
    label_list = list(LABEL_TO_ID[type_id].keys())
    label_idx = [TEST_LABEL_TO_ID[l] for l in label_list]
    # pred_probs 30개로 늘려주기
    pred_probs = []
    for idx, r in preds.iterrows():
        zeros = np.zeros(30)
        for jdx, p in zip(label_idx, eval(r['probs'])):
            zeros[jdx] = p
        binary_predictions.loc['id'==r['id'],'pred_label'] = r['pred_label'] 
        binary_predictions.loc['id'==r['id'],'probs'] = str(zeros)

binary_predictions.to_csv('/opt/ml/code/prediction/recent/depot-recent-final.csv')





            

    


    


# Re-assign from the predicted relation 
for k, v in binary_predictions.items():
    if v != '':continue 
    if k in semantic_preditions:
        binary_predictions[k] = semantic_preditions[k]
    else:
        binary_predictions[k] = 'no_relation'


predictions = []
for i in range(0, len(binary_predictions)):
    assert binary_predictions[i] != ''
    predictions.append(binary_predictions[i])

gold = []
data = open(y['gold_file'], 'r')
for d in data:
    d = d.strip()
    gold.append(d)

if not os.path.exists('saved_models/depot-all-recent/'):
    os.mkdir('saved_models/depot-all-recent/')

out_file = 'saved_models/depot-all-recent/predictions.txt'
out_f = open(out_file, 'w')
for i, p in enumerate(predictions):
    out_f.write('%d %s\n' % (i, p))

p, r, f1 = scorer.score(gold, predictions, verbose=True)
print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format('test',p,r,f1))


print("Evaluation ended.")


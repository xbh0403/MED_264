import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_adm = pd.read_csv('/Users/kuanhungyeh/Desktop/MED 264/ADMISSIONS.csv.gz', compression='gzip')
## Format the time variable
df_adm.ADMITTIME = pd.to_datetime(df_adm.ADMITTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DISCHTIME = pd.to_datetime(df_adm.DISCHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
df_adm.DEATHTIME = pd.to_datetime(df_adm.DEATHTIME, format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')


df_notes = pd.read_csv('/Users/kuanhungyeh/Downloads/NOTEEVENTS.csv.gz', compression='gzip')

df_notes = df_notes.sort_values(by=['SUBJECT_ID','HADM_ID','CHARTDATE'])
df_adm_notes = pd.merge(df_adm[['SUBJECT_ID','HADM_ID','ADMITTIME','DISCHTIME','ADMISSION_TYPE','DEATHTIME']],
                        df_notes[['SUBJECT_ID','HADM_ID','CHARTDATE', 'CHARTTIME', 'ISERROR','TEXT','CATEGORY']], 
                        on = ['SUBJECT_ID','HADM_ID'],
                        how = 'left')

## Format the time variable
df_adm_notes.ADMITTIME_C = df_adm_notes.ADMITTIME.apply(lambda x: str(x).split(' ')[0])
df_adm_notes['ADMITTIME_C'] = pd.to_datetime(df_adm_notes.ADMITTIME_C, format = '%Y-%m-%d', errors = 'coerce')
df_adm_notes['CHARTDATE'] = pd.to_datetime(df_adm_notes.CHARTDATE, format = '%Y-%m-%d', errors = 'coerce')

## Filter out a physician has identified this note as an error
df_adm_notes = df_adm_notes[df_adm_notes['ISERROR'].isna()]

## Filter out Discharge summary
df_adm_notes = df_adm_notes[df_adm_notes['CATEGORY']!='Discharge summary']

## Died within 30 days
df_adm_notes['OUTPUT_LABEL'] = df_adm_notes.apply(lambda row: 1 if (row['DEATHTIME'] - row['ADMITTIME']).days < 30 else 0, axis=1)

## result1 = df_adm_notes.groupby('OUTPUT_LABEL').size().reset_index(name='count')

### If Less than n days on admission notes (Early notes)
def less_n_days_data (df_adm_notes, n):
    
    df_less_n = df_adm_notes[((df_adm_notes['CHARTDATE']-df_adm_notes['ADMITTIME_C']).dt.total_seconds()/(24*60*60))<n]
    df_less_n=df_less_n[df_less_n['TEXT'].notnull()]
    #concatenate first
    df_concat = pd.DataFrame(df_less_n.groupby('HADM_ID')['TEXT'].apply(lambda x: "%s" % ' '.join(x))).reset_index()
    df_concat['OUTPUT_LABEL'] = df_concat['HADM_ID'].apply(lambda x: df_less_n[df_less_n['HADM_ID']==x].OUTPUT_LABEL.values[0])
    return df_concat

df_less_1 = less_n_days_data(df_adm_notes, 1)

## result2 = df_less_1.groupby('OUTPUT_LABEL').size().reset_index(name='count') # 4855 & 47891

## preprocess
import re
def preprocess1(x):
    y=re.sub('\\[(.*?)\\]','',x) #remove de-identified brackets
    y=re.sub('[0-9]+\.','',y) #remove 1.2. since the segmenter segments based on this
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    return y

def preprocessing(df_less_n):
    df_less_n['TEXT']=df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\n',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].str.replace('\r',' ')
    df_less_n['TEXT']=df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT']=df_less_n['TEXT'].str.lower()

    df_less_n['TEXT']=df_less_n['TEXT'].apply(lambda x: preprocess1(x))

    # Initialize an empty list to store data
    want_data = []

    from tqdm import tqdm

    for i in tqdm(range(len(df_less_n))):
        x = df_less_n.TEXT.iloc[i].split()
        n = int(len(x) / 318)

        for j in range(n):
            want_data.append({
                'TEXT': ' '.join(x[j * 318:(j + 1) * 318]),
                'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                'ID': df_less_n.HADM_ID.iloc[i]
            })

        if len(x) % 318 > 10:
            want_data.append({
                'TEXT': ' '.join(x[-(len(x) % 318):]),
                'Label': df_less_n.OUTPUT_LABEL.iloc[i],
                'ID': df_less_n.HADM_ID.iloc[i]
            })

    # Create a DataFrame from the list of dictionaries
    want = pd.DataFrame(want_data)

    return want
    
df_less_1_processed = preprocessing(df_less_1)

## result = df_less_1.groupby('OUTPUT_LABEL').size().reset_index(name='count')
## unique_subject_ids = df_less_1_processed['ID'].nunique() 
## unique_subject_ids #52506

## result3 = df_less_1.groupby('ID')['Label'].value_counts().unstack(fill_value=0).reset_index()
## count_label_1 = result3[result3[1] >= 1]['ID'].nunique()
## print(count_label_1) ## 4852 & 47702

thirtymortality_ID = df_less_1[df_less_1.OUTPUT_LABEL == 1].HADM_ID
not_thirtymortality_ID = df_less_1[df_less_1.OUTPUT_LABEL == 0].HADM_ID

# 80% training, 10% validation and 10% testing
id_val_test_t=thirtymortality_ID.sample(frac=0.2,random_state=1)
id_val_test_f=not_thirtymortality_ID.sample(frac=0.2,random_state=1)

id_train_t = thirtymortality_ID.drop(id_val_test_t.index)
id_train_f = not_thirtymortality_ID.drop(id_val_test_f.index)

id_val_t=id_val_test_t.sample(frac=0.5,random_state=1)
id_test_t=id_val_test_t.drop(id_val_t.index)

id_val_f=id_val_test_f.sample(frac=0.5,random_state=1)
id_test_f=id_val_test_f.drop(id_val_f.index)

# test if there is overlap between train and test, should return "array([], dtype=int64)"
(pd.Index(id_test_t).intersection(pd.Index(id_train_t))).values

id_test = pd.concat([id_test_t, id_test_f])
test_id_label = pd.DataFrame(data = list(zip(id_test, [1]*len(id_test_t)+[0]*len(id_test_f))), columns = ['id','label'])

id_val = pd.concat([id_val_t, id_val_f])
val_id_label = pd.DataFrame(data = list(zip(id_val, [1]*len(id_val_t)+[0]*len(id_val_f))), columns = ['id','label'])

id_train = pd.concat([id_train_t, id_train_f])
train_id_label = pd.DataFrame(data = list(zip(id_train, [1]*len(id_train_t)+[0]*len(id_train_f))), columns = ['id','label'])

#get train/val/test
df_less_1_train = df_less_1_processed[df_less_1_processed.ID.isin(train_id_label.id)]
df_less_1_val = df_less_1_processed[df_less_1_processed.ID.isin(val_id_label.id)]
df_less_1_test = df_less_1_processed[df_less_1_processed.ID.isin(test_id_label.id)]

## check proportiona of label
## df_less_1_train.Label.value_counts() 0: 105040 & 1: 13054
## df_less_1_val.Label.value_counts() 0: 13051 & 1: 1550
## df_less_1_test.Label.value_counts() 0: 13312 & 1: 1623

## write csv
df_less_1_train.to_csv('/Users/kuanhungyeh/Desktop/MED 264/day1_30mortality_train.csv')
df_less_1_val.to_csv('/Users/kuanhungyeh/Desktop/MED 264/day1_30mortality_val.csv')
df_less_1_test.to_csv('/Users/kuanhungyeh/Desktop/MED 264/day1_30mortality_test.csv')

## result3 = df_less_1_train.groupby('ID')['Label'].value_counts().unstack(fill_value=0).reset_index()
## count_label_1 = result3[result3[1] >= 1]['ID'].nunique()
## print(count_label_1) ## 4852 & 47702


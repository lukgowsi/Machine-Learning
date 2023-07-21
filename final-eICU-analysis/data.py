import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

def load_data(x_path):
    # Your code here
    df = pd.read_csv(x_path, index_col=False, low_memory=False)
    
    return df


def split_data(x, y, split=0.2):
    # Your code here
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=split, random_state=42)
    
    return train_x, train_y, test_x, test_y


def preprocess_x(df):
    # Your code here
    df = df.drop(df[df['cellattributevalue'] == 'feet'].index)
    df = df.drop(df[df['cellattributevalue'] == 'hands'].index)

    df['age'] = df['age'].replace("> 89", "90")
    withoutNull = df['age'].dropna()
    withoutNull = withoutNull.astype(int)
    df['nursingchartvalue'] = pd.to_numeric(df['nursingchartvalue'], errors='coerce')

    df['cellattributevalue'] = df['cellattributevalue'].replace("< 2 seconds", "normal")

    df = df.drop('offset', axis=1)
    df = df.drop('celllabel', axis=1)
    df = df.drop('labmeasurenamesystem', axis=1)
    df = df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1)

    grouped_data = df.groupby('patientunitstayid', group_keys=False)
    df = grouped_data.apply(lambda x: x.ffill().bfill())

    df['admissionheight'] = df['admissionheight'].fillna(df['admissionheight'].mean())
    df['admissionweight'] = df['admissionweight'].fillna(df['admissionweight'].mean())
    df['age'] = df['age'].fillna(withoutNull.mean())
    df['age'] = df['age'].astype(int)
    df['ethnicity'] = df['ethnicity'].replace("Native American", np.nan)
    df['ethnicity'] = df['ethnicity'].fillna(value = 'Other/Unknown') 
    df['gender'] = df['gender'].fillna(value = df['gender'].mode())
    df['cellattributevalue'] = df['cellattributevalue'].fillna(value = 'normal')

    labMeans = df.groupby(['patientunitstayid', 'labname'], group_keys=False)['labresult'].transform('mean')

    df['glucose'] = np.where(df['labname'] == 'glucose', labMeans, np.nan)
    df['ph'] = np.where(df['labname'] == 'pH', labMeans, np.nan)

    df[['glucose', 'ph']] = df.groupby('patientunitstayid', group_keys=False)[['glucose', 'ph']].apply(lambda x: x.fillna(x.mean()))


    nurseMeans = df.groupby(['patientunitstayid', 'nursingchartcelltypevalname'], group_keys=False)['nursingchartvalue'].transform('mean')

    df['heart rate'] = np.where(df['nursingchartcelltypevalname'] == 'Heart Rate', nurseMeans, np.nan)
    df['respiratory rate'] = np.where(df['nursingchartcelltypevalname'] == 'Respiratory Rate', nurseMeans, np.nan)
    df['gcs total'] = np.where(df['nursingchartcelltypevalname'] == 'GCS Total', nurseMeans, np.nan)
    df['o2 saturation'] = np.where(df['nursingchartcelltypevalname'] == 'O2 Saturation', nurseMeans, np.nan)
    df['non invasive bp systolic'] = np.where(df['nursingchartcelltypevalname'] == 'Non-Invasive BP Systolic', nurseMeans, np.nan)
    df['non invasive bp mean'] = np.where(df['nursingchartcelltypevalname'] == 'Non-Invasive BP Mean', nurseMeans, np.nan)
    df['non invasive bp diastolic'] = np.where(df['nursingchartcelltypevalname'] == 'Non-Invasive BP Diastolic', nurseMeans, np.nan)
    df['invasive bp diastolic'] = np.where(df['nursingchartcelltypevalname'] == 'Invasive BP Diastolic', nurseMeans, np.nan)
    df['invasive bp mean'] = np.where(df['nursingchartcelltypevalname'] == 'Invasive BP Mean', nurseMeans, np.nan)
    df['invasive bp systolic'] = np.where(df['nursingchartcelltypevalname'] == 'Invasive BP Systolic', nurseMeans, np.nan)

    df[['heart rate', 'respiratory rate', 'gcs total', 'o2 saturation', 'non invasive bp systolic', 'non invasive bp mean', 'non invasive bp diastolic', 'invasive bp diastolic', 'invasive bp mean', 'invasive bp systolic']] = df.groupby('patientunitstayid', group_keys=False)[['heart rate', 'respiratory rate', 'gcs total', 'o2 saturation', 'non invasive bp systolic', 'non invasive bp mean', 'non invasive bp diastolic', 'invasive bp diastolic', 'invasive bp mean', 'invasive bp systolic']].apply(lambda x: x.fillna(x.mean()))

    df['glucose'] = df['glucose'].fillna(df['glucose'].mean())
    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['heart rate'] = df['heart rate'].fillna(df['heart rate'].mean())
    df['respiratory rate'] = df['respiratory rate'].fillna(df['respiratory rate'].mean())
    df['gcs total'] = df['gcs total'].fillna(df['gcs total'].mean())
    df['o2 saturation'] = df['o2 saturation'].fillna(df['o2 saturation'].mean())
    df['non invasive bp systolic'] = df['non invasive bp systolic'].fillna(df['non invasive bp systolic'].mean())
    df['non invasive bp mean'] = df['non invasive bp mean'].fillna(df['non invasive bp mean'].mean())
    df['non invasive bp diastolic'] = df['non invasive bp diastolic'].fillna(df['non invasive bp diastolic'].mean())
    df['invasive bp diastolic'] = df['invasive bp diastolic'].fillna(df['invasive bp diastolic'].mean())
    df['invasive bp mean'] = df['invasive bp mean'].fillna(df['invasive bp mean'].mean())
    df['invasive bp systolic'] = df['invasive bp systolic'].fillna(df['invasive bp systolic'].mean())

    df = df.drop('labname', axis=1)
    df = df.drop('labresult', axis=1)
    df = df.drop('nursingchartcelltypevalname', axis=1)
    df = df.drop('nursingchartvalue', axis=1)
    
    # amt_patient_ids = df['patientunitstayid'].unique().size
    patientID = []
    for i in range(len(df)):
        row = df.iloc[i]
        if row['patientunitstayid'] not in patientID:
            patientID.append(row['patientunitstayid'])
    
    data = df.iloc[:len(patientID)]
    
    data = pd.get_dummies(data)
    
    return data

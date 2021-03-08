# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:39:13 2021

@author: Neha
"""

# get data from : http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz


# unzipping of zipped file - input data

import gzip
import shutil

with gzip.open('C:/Users/Neha/Desktop/flask/Clothing_Shoes_and_Jewelry.json.gz', 'rb') as f_in:
    with open('file.json', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        



# reading input and creating final dataset

import pandas as pd

df_reader = pd.read_json('C:/Users/Neha/Desktop/flask/file.json', lines = True, chunksize = 1000000 )

df = pd.DataFrame(columns=['overall','reviewText','summary'])

for chunk in df_reader:
    new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])
    new_df1 = new_df[new_df['overall'] == 5].sample(6000)
    new_df2 = new_df[new_df['overall'] == 4].sample(6000)
    new_df3 = new_df[new_df['overall'] == 3].sample(12000)
    new_df4 = new_df[new_df['overall'] == 2].sample(6000)
    new_df5 = new_df[new_df['overall'] == 1].sample(6000)
    
    df = pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5,df], axis = 0, ignore_index = True)
    new_df = None
    
print(df.shape)         # (1188000,3)
    
# saving the dataset
df.to_csv('balanced_review.csv', index = False)


#%%

#Import the necessary packages and load the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
filePath = "C:\\Users\\malakm\\Documents\\Gwu\\DataMining Intro\\Project 1\\LinkedIn_Data.csv"
dfOriginalData = pd.read_csv(filePath,encoding='latin1')
dfOriginalData.shape

#%%

# remove duplicates
dfOriginalData.drop_duplicates(subset =["avg_n_pos_per_prev_tenure", "avg_pos_len","m_urn","age"], keep = 'first', inplace = True) 
dfOriginalData.shape

#%%
# remove usless columns
dfOriginalData.drop(["beauty","beauty_female","beauty_male","blur","blur_gaussian","blur_motion","emo_anger","emo_disgust","emo_fear","emo_happiness","emo_neutral","emo_sadness","emo_surprise","ethnicity","face_quality","glass","head_pitch","head_roll","head_yaw","img","mouth_close","mouth_mask","mouth_open","mouth_other","skin_acne","skin_dark_circle","skin_health","skin_stain"], axis =1,inplace =True)
dfOriginalData.shape

dfOriginalData.to_csv("DataWODupCol.csv")
#%%

from linkedin_api import Linkedin

linkedin = Linkedin("minafawzi.us@gmail.com", "GWU@123$$")
results = linkedin.get_company('TT-Games-Ltd')
#temp = results["companyIndustries"][0]["localizedName"]
#print(temp)
print(results["companyIndustries"])

for x in dfOriginalData.shape[0]





#%%

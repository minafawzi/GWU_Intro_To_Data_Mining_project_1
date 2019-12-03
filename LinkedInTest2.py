
#%%

#Import the necessary packages and load the dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
filePath = "C:\\Users\\HP ELITEBOOK\\Documents\\dataming\\GWU_Intro_To_Data_Mining_project_1\\Linkedin_Data.csv"
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

#for x in dfOriginalData.shape[0]


#%%
dfOriginalData.head(5)


#%%

# Ghadeer code


#cleaning the data 
#remove people who are younger than 18 
dfOriginalData= dfOriginalData[ dfOriginalData['age'] > 17 ]

# Histogram the age in general 
plt.style.use('seaborn-deep')
age= dfOriginalData['age'] 
bins = 7
plt.hist(age, bins, alpha=0.7, edgecolor='black', linewidth=1 )
plt.xlabel('Age of Employee')
plt.ylabel('Frequency')


# Histogram the age/gender 

DF_Male= dfOriginalData[ dfOriginalData['gender']== "Male" ]
Age_M= DF_Male["age"]
DF_Female= dfOriginalData[ dfOriginalData['gender']== "Female" ]
Age_F= DF_Female["age"]

plt.style.use('seaborn-deep')

plt.hist([Age_M,Age_F],  label=['Male','Female'], alpha=0.9 )
plt.show()


# showing how the data is biased since there is only 23% female 
Gender_count= [DF_Male['gender'].count(), DF_Female['gender'].count()]
DF_Gender_count= pd.DataFrame(Gender_count)
labels=['Male', 'Female']
colors = ['lightskyblue', 'lightcoral']
DF_Gender_count.plot(kind='pie' , subplots=True, colors=colors , autopct='%1.1f%%', shadow=True, startangle=200, labels=labels )
plt.show()

# analyzing the Female Data (DF_Female)
# number of position

One_p= DF_Female[DF_Female["n_pos"]== 1 ]
two_p= DF_Female[DF_Female["n_pos"]== 2 ]
three_p= DF_Female[DF_Female["n_pos"]== 3 ]
#four_p= DF_Female[DF_Female["n_pos"]== 4]
#five_p= DF_Female[DF_Female["n_pos"]== 5 ]
rest_p= DF_Female[DF_Female["n_pos"] > 3 ]
pos_list= [One_p['n_pos'].count(), two_p['n_pos'].count(), three_p['n_pos'].count(), rest_p['n_pos'].count() ]
DF_pos_F= pd.DataFrame(pos_list)
labels=['One position', 'Two position', 'Three position', 'Four or more' ]
colors = ['lightcoral', 'yellowgreen', 'gold', 'lightskyblue', 'aqua', 'thistle']
DF_pos_F.plot(kind='pie' ,subplots=True,  colors=colors , autopct='%1.1f%%', shadow=True, startangle=200, labels=labels )
plt.show()


# analyzing the Male Data (DF_Male)
# number of position

One_p= DF_Male[DF_Male["n_pos"]== 1 ]
two_p= DF_Male[DF_Male["n_pos"]== 2 ]
three_p= DF_Male[DF_Male["n_pos"]== 3 ]
#four_p= DF_Female[DF_Female["n_pos"]== 4]
#five_p= DF_Female[DF_Female["n_pos"]== 5 ]
rest_p= DF_Male[DF_Male["n_pos"] > 3 ]

pos_list2= [One_p['n_pos'].count(), two_p['n_pos'].count(), three_p['n_pos'].count(), rest_p['n_pos'].count() ]
DF_pos_M= pd.DataFrame(pos_list2)

labels=['One position', 'Two position', 'Three position', 'Four or more' ]
colors = [ 'lightskyblue', 'yellowgreen', 'gold', 'lightcoral', 'thistle']
DF_pos_M.plot(kind='pie' , subplots=True, colors=colors , autopct='%1.1f%%', shadow=True, startangle=200, labels=labels )
plt.show()


# There is no significant differnce of the average tenure among different ethencity 

N_English= dfOriginalData[dfOriginalData['nationality']== "celtic_english" ]
N_european= dfOriginalData[dfOriginalData['nationality']== "european" ]
N_south_asian= dfOriginalData[dfOriginalData['nationality']== "south_asian" ]
N_hispanic= dfOriginalData[dfOriginalData['nationality']== "hispanic" ]
N_nordic= dfOriginalData[dfOriginalData['nationality']== "nordic" ]
N_african = dfOriginalData[dfOriginalData['nationality']== "african" ]
N_greek= dfOriginalData[dfOriginalData['nationality']== "greek" ]
N_mean=[N_English["tenure_len"].mean() , N_european["tenure_len"].mean(), N_hispanic["tenure_len"].mean(),N_south_asian["tenure_len"].mean(), N_nordic["tenure_len"].mean(), N_african["tenure_len"].mean(), N_greek["tenure_len"].mean() ]

# Boxplot of tenure_len grouped by ethencity

# dropping the jewish and Musilm since it's religion not ethnicity nor nationality 
Nationality_DF= dfOriginalData[ (dfOriginalData['nationality'] != "jewish") & (dfOriginalData['nationality'] != "muslim")]


Nationality_DF.boxplot(column= 'tenure_len' , by= "nationality", showfliers=False  )


plt.show() 



#There is insignificant differnce of the average tenure among different gender

DF_Male['tenure_len'].mean()
DF_Female['tenure_len'].mean()
Gender_avg_tenure= [DF_Male['tenure_len'].mean(), DF_Female['tenure_len'].mean() ]

dfOriginalData.boxplot(column= 'tenure_len' , by= "gender", showfliers=False )

plt.show() 



# Age and tenure ( as people are getting older the avg tenure increase  )
# significant trend 
# age between 18-23 / avg 619

Group1= dfOriginalData[ dfOriginalData['age']< 24] 

# age between 24-30 / avg 741
Group2= dfOriginalData[ (dfOriginalData['age'] > 23) & (dfOriginalData['age']< 30) ]

#age between 31- 40 / avg 835
Group3= dfOriginalData[ (dfOriginalData['age'] > 30 ) & (dfOriginalData['age']< 41) ]

#age between 41- 50 / avg 981
Group4= dfOriginalData[ (dfOriginalData['age'] > 40 ) & (dfOriginalData['age']< 51) ]

#age between 51- 80 / avg 1185
Group5= dfOriginalData[ (dfOriginalData['age'] > 50 ) & (dfOriginalData['age']< 82) ]

# plot 

Age_G= [Group1["age"].count(), Group2["age"].count(),Group3["age"].count(), Group4["age"].count(), Group5["age"].count()  ]
#DF_summary2= pd.DataFrame(survived_M["id"].count(), Dead_M["id"].count(),survived_F["id"].count(), Dead_F["id"].count() )
Age_Groups= pd.DataFrame(Age_G)
labels=['18-23 YO', '24-30 YO', '31-40 YO', '41-50YO' , '51-80 YO']
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'thistle']
Age_Groups.plot(kind='pie' , subplots=True, colors=colors , autopct='%1.1f%%', shadow=True, startangle=200, labels=labels )
plt.show()



#%%

#Lou's EDA

dfOriginalData.describe()


# %%
num_bins = 10
plt.hist(dfOriginalData['age'], num_bins, normed=1, facecolor='blue', alpha=0.5)
plt.show()
#Age is evenly distributed.  Mean is 44 yrs

# %%
# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# %%
plotPerColumnDistribution(dfOriginalData, 10, 5)

#more male than female.  # heavily celtic english

# %%
#avg age for the ethnic groups
#nunique - sum values per Gender and Type groups
avg_age = dfOriginalData.groupby(['gender','nationality'],as_index = False).agg({'age': 'mean'})
print (avg_age)

# %%
#gender by Nationality
figure = plt.figure(figsize=(15,8))
plt.hist([dfOriginalData[dfOriginalData['gender']=='Female']['nationality'], dfOriginalData[dfOriginalData['gender']=='Male']['nationality']], stacked=True, bins=30, label=['Female','Male'])
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.legend()




# %%
#nationality by avg age
import seaborn as sns
sns.set()
avgage = sns.barplot(x="age", y="nationality", hue="gender", data=avg_age)

# %%

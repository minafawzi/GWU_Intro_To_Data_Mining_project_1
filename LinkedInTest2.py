
#%% [markdown]
#  The following code goes thru analysis of the linkedin data
#
# # Pre-processing and data cleaning
# - First we load the data
# - Clean the data from duplicates
# - Drop unncessary columns that will not be used for the analysis 
# - Create visuals to understand the data dimensions and various features

#%%

#Import the necessary packages and load the dataset
# using the Latin1 encoding to process some special characters in the company names

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
# remove useless columns
dfOriginalData.drop(["beauty","beauty_female","beauty_male","blur","blur_gaussian","blur_motion","emo_anger","emo_disgust","emo_fear","emo_happiness","emo_neutral","emo_sadness","emo_surprise","ethnicity","face_quality","glass","head_pitch","head_roll","head_yaw","img","mouth_close","mouth_mask","mouth_open","mouth_other","skin_acne","skin_dark_circle","skin_health","skin_stain"], axis =1,inplace =True)
dfOriginalData.shape

dfOriginalData.to_csv("DataWODupCol.csv")

#%%
# histogram for male vs female age 
uniqueProfiles = dfOriginalData[dfOriginalData["n_prev_tenures"]==1]
ages_male   = uniqueProfiles[dfOriginalData["gender"] == "Male"]
ages_female = uniqueProfiles[dfOriginalData["gender"] == "Female"]

ages_male   = ages_male["age"]
ages_female = ages_female["age"]

plt.hist(ages_male, label='Ages_male',edgecolor='black', linewidth=1)
plt.hist(ages_female, label='Ages_female',edgecolor='black', linewidth=1)
plt.show()

#%%
fuzzyFollower = uniqueProfiles["n_followers"] + np.random.normal(0,2, size=len(uniqueProfiles["n_followers"]))
plt.plot(uniqueProfiles["age"], fuzzyFollower, 'o', markersize=4, alpha = 1)
plt.ylabel('Number of followers')
plt.xlabel('Age')
plt.show()

#%%
fuzzyage = uniqueProfiles["age"] + np.random.normal(0,1, size=len(uniqueProfiles["age"]))
plt.plot(fuzzyage, fuzzyFollower, 'o', markersize=4, alpha = 1)
plt.ylabel('Number of followers')
plt.xlabel('Age')
plt.show()

#%%
#%% [markdown]
#  The following code is peforming the linkedIn API connection
#  the API is to be installed first using: pip install linkedin-api
#  information about the wrapper is here: https://pypi.org/project/linkedin-api/

# # LinkedIn API connection 
# - Load the package
# - Initantiate the object with the credentials
# - Create a list of unique company names
# - Convert the company name to the searachable format (i.e. replace spaces with dash)
# - Ping the API for each company name to retrieve the industry 
# - If industry is Nan then we will try to other naming combination that might match the searchable pattern

#%%
# the below code connects to the LinkedIn API and iterate thru the unique company name lists
# the code will then populate the company industry dataframe that contains only three columns: 
# 1- companyName : actual name in the dataset
# 2- companyNameLinkedin: name used to search the linkedin 
# 3- companyIndustry: industry retrieved from the API


# After doing basic string functions, the code invokes tryCompanyIndustry that will try various options and handles 
# the exception in the API call

from linkedin_api import Linkedin
linkedin = Linkedin("minafawzi.us@gmail.com", "GWU@123$$")

#try for 50 before full 
# There was some performance issues we discovered with this functions
# so we split the processing to chunks of 500 or more records per connection then we renew the connection
#companyData = dfOriginalData["c_name"].head(50).unique()
companyData = dfOriginalData["c_name"].unique()[1600:1900]
companyData = companyData [~pd.isnull(companyData)]
companyNameList = []
companyIndustriesList = []
companyNameLinkedinList = []
companyNameLinkedin=""
companyIndustry=""
companyName=""

for i in companyData:
    companyName = i
    companyNameLinkedin = companyName
    if (companyName.find("(") != -1):
        theIndex = companyName.find("(")
        companyNameLinkedin = companyName[0:theIndex].rstrip()
    companyNameLinkedin = companyNameLinkedin.replace(" ","-").replace(",","")
    print("checking for",companyNameLinkedin,"...")
    companyIndustry =tryCompanyIndustry(linkedin,companyNameLinkedin)
    
    companyNameList.append(companyName)
    companyIndustriesList.append(companyIndustry)
    companyNameLinkedinList.append(companyNameLinkedin)

print("**********Done**************")

#%%
# once we capture company functions we save them in various csv files that then will be combined to have a master
# mapping document
Temp = {"CompanyName":companyNameList,"companyNameLinkedin":companyNameLinkedinList,"companyIndustries":companyIndustriesList}
DFTemp = pd.DataFrame(Temp)
DFTemp.to_csv("1400_1600.csv")
# %%
# the first try is to see if the company name contains any digits, this is where the space or dash after the digits
# will be deteted and then we re-call the API to see if it is working

# the second try is to see if the first name in the company name to be used alone, i.e. Microsoft India might now work but 
# Microsoft only will work
  """
  this function calls the getCompanyIndustry function and if the return is Nan then it tries 2 more variation of the name
  the variations were analyzed with some try an error and manual search in linkedin to see how it is made
  
  :param linkedinObj: this is the LinkedIn API object that contains the loogging credentials
  :param string companyName: the company industry that was returned from the API 
  :return: company industry 
  """

from linkedin_api import Linkedin
def tryCompanyIndustry(linkedinObj, companyName):
    industry = getCompanyIndustry(linkedinObj,companyName)
    if(industry == "Nan" and hasNumbers(companyName)):
        print(companyName, "   Has numbers..")
        if(companyName.find("0-")!=-1):
            print("Found 0-")
            companyName = companyName.replace("0-","0")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("1-")!=-1):
            print("Found 1-")
            companyName = companyName.replace("1-","1")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("2-")!=-1):
            print("Found 2-")
            companyName = companyName.replace("2-","2")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("3-")!=-1):
            print("Found 3-")
            companyName = companyName.replace("3-","3")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("4-")!=-1):
            print("Found 4-")
            companyName = companyName.replace("4-","4")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("5-")!=-1):
            print("Found 5-")
            companyName = companyName.replace("5-","5")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("6-")!=-1):
            print("Found 6-")
            companyName = companyName.replace("6-","6")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("7-")!=-1):
            print("Found 7-")
            companyName = companyName.replace("7-","7")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("8-")!=-1):
            print("Found 8-")
            companyName = companyName.replace("8-","8")
            industry = getCompanyIndustry(linkedinObj,companyName)
        if(companyName.find("9-")!=-1):
            print("Found 9-")
            companyName = companyName.replace("9-","9")
            industry = getCompanyIndustry(linkedinObj,companyName)
    if(industry == "Nan" and len(companyName.split("-"))>1):
        print("trying first name only...")
        industry = getCompanyIndustry(linkedinObj,companyName.split("-")[0])
    return industry
    

#%%

  """
  Checks if there is any digits in a string
  
  :param string inputString: input string to be checked
  :return: bool  
  """

def hasNumbers(inputString):
...     return any(char.isdigit() for char in inputString)


#%%

# the actuak function doing the API call
# return JSON (if not nan) will then be searched for the specific location where indutsry is mentioned
  """
  This is the actual function that conduct the API call
  
  :param linkedinObj: this is the LinkedIn API object that contains the loogging credentials
  :param string companyName: the company industry that was returned from the API 
  :return: company industry 
  """
from linkedin_api import Linkedin
def getCompanyIndustry(linkedinObj, companyName):
    
    try:
        result = linkedinObj.get_company(companyName)["companyIndustries"][0]["localizedName"]
        return result
    except:
        return "Nan"

#%%
dfOriginalData.head(5)

# %%[markdown]
## company industries
# based on the industries table we will trim them down to one of the following industry super groups:
#- "Computer, Telecom and IT"
#"- Financial Services and Banking"  
#- "Online Media","Writing & Editing"
#- "Enegry and Chemicals"
#- "Gov, Law and NGOs"
#- "Retail"
#- "Medical Services"
#- "Aviation, Defense and Maritime"
#- "Consulting and outsourcing"
#- "Education"
#- "Manufacturing"
#- "Toursim, hotels and restaurants"
#- "Food & Beverages"
#- "Logistics and Transportation"

# all other industries will be referred to as "Other"
#%%
filePath = "C:\\Users\\malakm\\Documents\\Gwu\\DataMining Intro\\Project 1\\All_CompanyIndustries.csv"
companyIndustriesData = pd.read_csv(filePath,encoding='latin1')
companyIndustriesData.shape
#uniqueIndustries = companyIndustriesData["companyIndustries"].unique()

companyIndustriesData["companyIndustries"].replace(["Internet","Computer Software","Computer Hardware","Computer Networking","Computer & Network Security","Information Technology & Services","Information Services","Computer Software","Computer Games","Consumer Electronics","Telecommunications","Wireless","Information","Semiconductors"], "Computer, Telecom and IT", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Banking","Financial Services","Accounting","Investment Management","Insurance","Venture Capital & Private Equity","Capital Markets","Investment Banking"], "Financial Services and Banking", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Printing","Broadcast Media","Publishing","Photography","Media Production","Newspapers","Graphic Design","Entertainment","Music","Motion Pictures & Film","Animation","Translation & Localization","Online Media","Writing & Editing"], "Media", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Renewables & Environment","Chemicals","Oil & Energy","Mining & Metals"], "Enegry and Chemicals", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Environmental Services","Legal Services","Non-profit Organization Management","Security & Investigations","Law Practice","Philanthropy","Government Administration","Civic & Social Organization","Public Safety","Public Policy","Government Relations","Political Organization","International Affairs","Judiciary","International Trade & Development","Law Enforcement"], "Gov, Law and NGOs", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Consumer Services","Business Supplies & Equipment","Wholesale","Luxury Goods & Jewelry","Retail","Consumer Goods","Cosmetics","Wine & Spirits","Consumer Services","Apparel & Fashion","Furniture","Sporting Goods","Individual & Family Services","Tobacco"], "Retail", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Hospital & Health Care","Pharmaceuticals","Medical Device","Medical Practice","Mental Health Care","Biotechnology"], "Medical Services", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Aviation & Aerospace","Airlines/Aviation","Defense & Space","Maritime"], "Aviation, Defense and Maritime", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Program Development","Design","Outsourcing/Offshoring","Executive Office","Market Research","Management Consulting","Marketing & Advertising","Staffing & Recruiting","Human Resources","Architecture & Planning","Think Tanks"], "Consulting and outsourcing", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Higher Education","Primary/Secondary Education","Education Management","Research","E-learning","Professional Training & Coaching"], "Education", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Gambling & Casinos","Hospitality","Restaurants","Leisure, Travel & Tourism"], "Toursim, hotels and restaurants", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Construction","Building Materials","Mechanical Or Industrial Engineering","Machinery","Civil Engineering","Electrical & Electronic Manufacturing","Automotive","Industrial Automation"], "Manufacturing", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Dairy","Food Production","Farming","Food & Beverages"], "Food & Beverages", inplace=True)  
companyIndustriesData["companyIndustries"].replace(["Package/Freight Delivery","Logistics & Supply Chain","Transportation/Trucking/Railroad","Packaging & Containers"], "Logistics and Transportation", inplace=True)  

for ind in companyIndustriesData.index:
    temp = companyIndustriesData["companyIndustries"][ind]
    print (ind)
    if(temp =="Computer, Telecom and IT"):
        #print("match ","Computer, Telecom and IT" )
        continue
    elif(temp =="Financial Services and Banking"):
        #print("match ","Finance" )
        continue
    elif(temp =="Online Media Writing & Editing"):
        #print("match ","Media" )
        continue
    elif(temp =="Enegry and Chemicals"):
        #print("match ","Energy" )
        continue
    elif(temp =="Gov, Law and NGOs"):
        #print("match ","Gov" )
        continue
    elif(temp =="Retail"):
        #print("match ","Retail" )
        continue
    elif(temp =="Medical Services"):
        #print("match ","Medical" )
        continue
    elif(temp=="Aviation, Defense and Maritime"):
        #print("match ","Aviation" )
        continue
    elif(temp =="Consulting and outsourcing"):
        #print("match ","Consult" )
        continue
    elif(temp =="Education"):
        #print("match ","Edu" )
        continue
    elif(temp =="Manufacturing"):
        #print("match ","Manufacturing" )
        continue
    elif(temp=="Toursim, hotels and restaurants"):
        #print("match ","Toursim" )
        continue
    elif(temp =="Food & Beverages"):
        #print("match ","Food" )
        continue
    elif(temp =="Logistics and Transportation"):
        #print("match ","Logistics" )
        continue
    else:
        #print("NO NO NNO match ","Logistics" )
        companyIndustriesData["companyIndustries"][ind] = "Other"
        continue

# %%
# return the industry lookup from the Dataframe holding company industry per company name
"""
  :param companyIndustriesData: the dataframe for the company industry
  :param string companyName: the company name to be used in the lookup 
  :return: company industry from the dataframe
 """
def CompanyIndustryLookup(companyIndustriesData, companyName):
    found =0
    for index, i in companyIndustriesData.iterrows():
        if(i["CompanyName"] ==companyName ):
            found =1
            #print("Found ->",companyName )
            #print(i["companyIndustries"])
            return i["companyIndustries"]
    if(found == 0):
        return "Nan"

#%%
#companyIndustriesData.to_csv("companyIndustriesData.csv")
# must run the CompanyIndustryLookup() function first
# this will populate the industry inside the original dataframe
for index, row in dfOriginalData.head(12000).iterrows():
    dfOriginalData.set_value (index,"Industry",CompanyIndustryLookup(companyIndustriesData,row["c_name"]))

print("************Done*************")     
#%%
#now we have the original Dataset with a newly added column called Industry
# we will first drop all empty rows where industry was not populated
dfOriginalData2 = dfOriginalData
dfOriginalData2 = dfOriginalData2[dfOriginalData2["Industry"]!="Nan"]
dfOriginalData2 = dfOriginalData2[pd.notnull(dfOriginalData2["Industry"])]
dfOriginalData2.shape

# then we conduct various analysis grouped by the industry 

test1 = dfOriginalData2["age"].groupby(dfOriginalData2["Industry"])
#test1.mean().plot.bar()
test1.count().plot.bar()

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
num_bins = 20
plt.hist(dfOriginalData['age'], num_bins, facecolor='blue', alpha=1)
plt.xlabel('Age of Employee')
plt.ylabel('Frequency')
plt.show()
#Age is evenly distributed.  Mean is 44 yrs
#%%
import seaborn as sns
sns.set()
sns.distplot(dfOriginalData['age'],kde = False)

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
plotPerColumnDistribution(dfOriginalData, 5, 5)

#more male than female.  # heavily celtic english

#%%
# bar the nationality in general 
my_tab = pd.crosstab(index = dfOriginalData["nationality"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab.plot.bar()
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
#%%

## correlation and heatmap
# reload the original data again but without dropping the headshot columns to analyze smiling

from __future__ import print_function
%matplotlib inline
import matplotlib.pyplot as plt
import scipy
import os
import sys

filePath = "C:\\Users\\HP ELITEBOOK\\Documents\\dataming\\GWU_Intro_To_Data_Mining_project_1\\Linkedin_Data.csv"
dfOriginalData = pd.read_csv(filePath,encoding='latin1')
dfOriginalData.shape

#%%

df = dfOriginalData[['age','ethnicity','gender','mouth_close','smile','n_followers']]
#%%
df.corr(method='kendall')

#%%
plt.figure(figsize=(7,7))
seaborn.heatmap(df.corr(), annot=True, cmap= 'coolwarm')

#%%

df_pairplot = df.dropna()
seaborn.pairplot(df_pairplot, height=1.5)

#%%

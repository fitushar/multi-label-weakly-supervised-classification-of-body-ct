"""
@author: Fakrul Islam Tushar (fakrulislam.tushar@duke.edu)

© copyright 2020 - 2021 Duke University
-This program is free software and a property of Duke University:
you can redistribute it and/or modify for non-commercial uses only.
-This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""

# import pandas as pd
import numpy as np
import glob
import os
import re
from nltk.tokenize import sent_tokenize
import pickle
import csv
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from RBA_Kidneys_Config import*


###############---User_input-###################

Main_Report_CSV=REPORT_CSV
organ_dictonary="kidney_dict.pickle"
common_dis_list =LIST_FOR_OVERLAP_STATISTICS
Disease_Name_and_Number_Decending_CSV_NAME="Disease_Name_and_Number_Decending.csv"
Disease_count_threshold=100

#############----User_input_End---@@@@######

#loading the main report list
mylist = pd.read_csv(
        Main_Report_CSV,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
report_list=mylist

### Directory
pickle_in = open(organ_dictonary,"rb")
liver_dict = pickle.load(pickle_in)

for dis in common_dis_list:
    cases = liver_dict[dis]
    print('{}--#{}'.format(dis,len(cases)))


name_list=[]
number_list=[]

for i in range(0,len(common_dis_list)):
    name=common_dis_list[i]
    name_list.append(name)
    cases = liver_dict[name]
    number_list.append(len(cases))

    #print('{}--#{}'.format(name_list[i],number_list[i]))


### Saving the Sorted Excel file

Inf0_data=pd.DataFrame(list(zip(name_list,number_list)),
columns=['Disease_name','Number_of_cases'])
Disease_Name_and_Number_Decending= Inf0_data.sort_values(by ='Number_of_cases',ascending=False)
Disease_Name_and_Number_Decending.to_csv(Disease_Name_and_Number_Decending_CSV_NAME, encoding='utf-8', index=False)

Disease_Name_and_Number_Decending_CSV= pd.read_csv(
        Disease_Name_and_Number_Decending_CSV_NAME,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()


###Comparing with the threshold and give the disease numbers.

Greater_than_200_disease=[]
print("-----------------@@@@@@@@@@------------------------")
print("-----------------Top Diseases----------------------")
print("-----------------@@@@@@@@@@------------------------")
for i in range (0, len(Disease_Name_and_Number_Decending_CSV)):
    Disease_number=int(Disease_Name_and_Number_Decending_CSV[i][1])

    if (Disease_number >= Disease_count_threshold):
        print(Disease_Name_and_Number_Decending_CSV[i][0],Disease_Name_and_Number_Decending_CSV[i][1])
        Greater_than_200_disease.append(Disease_Name_and_Number_Decending_CSV[i][0])


###Writing the csv files for the top Diseases

print("-----------------@@@@@@@@@@------------------------")
print("----Writing the csv files for the top Diseases----")
print("-----------------@@@@@@@@@@------------------------")

def Writting_disease_CSV(lst,main_lst,Name):
    """
    Description: This Function will write the report number and report text in a  excel file.

    Inputs:
         list= The number you get from the dictonary
         main_lst= the Main Report CSV
         Name= SAVE CSV file name.

    Output: xxxx.csv
    """
    cases_list=[]
    txt_list=[]
    for i in range(0,len(main_lst)):
        sub=main_lst[i][0]
        txt = main_lst[i][2]
        if sub in lst:
            cases_list.append(sub)
            txt_list.append(txt)
    Inf0_data=pd.DataFrame(list(zip(cases_list,txt_list)),
    columns=['Report_id','Report'])
    Inf0_data.to_csv(Name, encoding='utf-8', index=False)
    print('total disease case={}--done Saving {}'.format(len(cases_list),Name))
    return Inf0_data


for dis in Greater_than_200_disease:
    if dis=='surgically\s+absent':
        name= 'surgically_absent.csv'
        cases = liver_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    elif dis=='fatty\\s+infiltr':
        name= 'fatty_infiltr.csv'
        cases = liver_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    else:
        name=dis+'.csv'
        cases = liver_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
        #print('Disease-->>{}---Number of Cases #{}'.format(dis,len(cases)))



print("-----------------@@@@@@@@@@------------------------")
print("------------Overlap between the Diseases-----------")
print("-----------------@@@@@@@@@@------------------------")

def overall(csv_1,csv_2):
    """
    Description: This Function Prints the Overlap between two insputed excel files.

    Inputs:
         csv_1= csv of report list for disease 1
         csv_2= csv of report list for disease 2


    Output: xxxx.csv
    """
    a_list=[]
    b_list=[]

    a_csv=pd.read_csv(csv_1,dtype=object, keep_default_na=False,na_values=[]).as_matrix()
    for i in range(len(a_csv)):
        a=a_csv[i][0]
        a_list.append(a)

    b_csv=pd.read_csv(csv_2,dtype=object, keep_default_na=False,na_values=[]).as_matrix()
    for i in range(len(b_csv)):
        b=b_csv[i][0]
        b_list.append(b)

    csv_1_d_name=csv_1.split('.csv')
    csv_2_d_name=csv_2.split('.csv')



    a_b_overlap=[]
    for sub in a_list:
        if sub in b_list:
            a_b_overlap.append(sub)

    print('Overlap between {} and {}= {}'.format(csv_1_d_name[0],csv_2_d_name[0],len(a_b_overlap)))
    n_l=len(a_b_overlap)
    return n_l

lm_list=[]

Greater_than_200_disease = [word.replace('surgically\\s+absent','surgically_absent') for word in Greater_than_200_disease]
Greater_than_200_disease = [word.replace('fatty\\s+infiltr','fatty_infiltr') for word in Greater_than_200_disease]

for i in range (0, len(Greater_than_200_disease)):
    print('-----------------@@@-{}-@@@@---------------'.format(Greater_than_200_disease[i]))
    lm_list2=[]
    for j in range(0,len(Greater_than_200_disease)):

        first_csv=Greater_than_200_disease[i]+'.csv'
        Second_csv=Greater_than_200_disease[j]+'.csv'

        im=overall(first_csv,Second_csv)
        lm_list2.append(im)

    lm_list.append(lm_list2)



print("-------------Confussion Matrix-----------")
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(lm_list, index = [i for i in Greater_than_200_disease],
                  columns = [i for i in Greater_than_200_disease])
df_cm.to_csv("400kKidney_diagnoses_OverlapTrain.csv", encoding='utf-8', index=False)

plt.figure(figsize = (70,70))
sn.heatmap(df_cm, annot=True,fmt="d")
plt.savefig('Kidney_diagnoses_OverlapTrian.png')

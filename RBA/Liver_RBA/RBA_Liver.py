###############
"""
@author: Fakrul Islam Tushar (fakrulislam.tushar@duke.edu)

© copyright 2020 - 2021 Duke University
-This program is free software and a property of Duke University:
you can redistribute it and/or modify for non-commercial uses only.
-This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""

###################import libraries#########################

import pandas as pd
import numpy as np
import glob
import os
import re
from nltk.tokenize import sent_tokenize
import csv
import pickle
from RBA_Liver_Config import*

####################Reading the Report CSV###################
## Reading the Report CSV
mylist = pd.read_csv(REPORT_CSV,dtype=object,keep_default_na=False,na_values=[]).as_matrix()
########################################################
## Making the Split_list
'''split_list is a list that has basically the reports in order
   (subid, anythingbeforeFindings, Findings, Impressions)
   #split_list[i][0]= Report Id,
   #split_list[i][1]= everything Before Finding
   #split_list[i][2]= Finding
   #split_list[i][3]= Impression
'''
report_list=mylist
split_list = []
for i in range(len(report_list)):
    num = report_list[i][0]
    txt1 = report_list[i][2]
    txt3 = re.sub(r"\.\.\.", ".",txt1) #||-- Correcting double . issue
    txt = re.sub(r"\.\.", ".",txt3) #||-- Correcting double . issue
    ## Getting the Finding and Impression Sections
    m1 = re.match(r"(.*)Findings:(.*)Impression:(.*)", txt)
    m2 = re.match(r"(.*)FINDINGS:(.*)IMPRESSION:(.*)", txt)
    m3 = re.match(r"(.*)Findings:(.*)IMPRESSION:(.*)", txt)
    m4 = re.match(r"(.*)FINDINGS:(.*)Impression:(.*)", txt)
    #                       0            1            2            3   4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54
    if m1 is not None:
        split_list.append([num, m1.group(1), m1.group(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif m2 is not None:
        split_list.append([num, m2.group(1), m2.group(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif m3 is not None:
        split_list.append([num, m3.group(1), m3.group(2), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif m4 is not None:
        split_list.append([num, m4.group(1), m4.group(2),0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    else:
        print("wrong")
organ_try_list=[]
disease_list=[]
############################# Functions #############################################
##Upading the Dictonary
def update_dict(thedict, key_a, key_b, val):
    adic = thedict.keys()
    if key_a in adic:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

## Diseased Term
def search_dis(dis_caught_flag, dis_name, organ_list, report, organ_dis_dict, write_dict, sentence_dis_dict, is_specific=0):
    ##Subject Id
    sub = report[0]

    ###Adding the Finding the Impression
    #a=report[2].lower()
    a=report[2]
    sentence_list = sent_tokenize(a)

    organ_sub_re = r''
    for re_tok in organ_list[2]:
        if re_tok=='hepatic':
            #organ_sub_re = organ_sub_re + '\s*' + re_tok+'*' + r'|' + re_tok.capitalize()+'*' + r'|'
            organ_sub_re = organ_sub_re + '\s*' + re_tok+'*' + r'|' + re_tok.capitalize() +'*'+ r'|'
            #print(organ_sub_re)
        else:
            organ_sub_re = organ_sub_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
            #print(organ_sub_re)
    organ_sub_re = organ_sub_re[:-1]
    #print(organ_sub_re)

    dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()
    #dis_re= r'\s*' + dis_name+'*' + r'|' + dis_name.capitalize()+'*'

    deny_re = r'\sno\s|No\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|\snofocal|' \
    r'\snegative\s|Negative\s|with\s+regards\s+to|\showever\s|' \
    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution'

    if dis_name == 'mass':
        deny_re = r'\sno\s|No\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|' \
    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|\smass\s+effect|Mass\s+effect'

    if dis_name=='dilat':
        dis_re= r'\s*' + dis_name+'*' + r'|' + dis_name.capitalize()+'*'
        deny_re = r'\sno\s|No\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\s*nointra*|Nointra*|\s*Nointra*|\s*noextra*|Noextra*|\s*Noextra*|\sisno\s|Noevidence|\snoevidence\s|\snofocal|' \
    r'\snegative\s|Negative\s|with\s+regards\s+to|\showever\s|' \
    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution'




    ### Adding wild cards for Surgically Absent
    if dis_name=='surgically\s+absent':
        dis_re=r'\s*surgically\s+absent*|\s*Surgically\s+absent*|\s*surgicallyabsent*|\s*Surgicallyabsent*|\s*surgically\s+removed*|\s*Surgically\s+removed*|\s*surgicallyremoved*|\s*Surgicallyremoved*|\s*absent*|\s*Absent*'
    ##---Fatty wild card
    if dis_name=='fatty\s+infiltr':
        dis_re=r'\s*fatty\s+infiltr*|\s*Fatty\s+infiltr*|\s*fattyinfiltr*|\s*Fattyinfiltr*'

    #MAKING GallStone AS wild Card*
    if dis_name=='gallstone':
        dis_re= r'\s*' + dis_name+'*' + r'|' + dis_name.capitalize()+'*'
    #MAKING lesion AS wild Card*
    if dis_name=='lesion':
        dis_re= r'\s*' + dis_name+'*' + r'|' + dis_name.capitalize()+'*'

    ###----Making the keys of dictonary
    caught = 0
    if dis_name not in organ_dis_dict.keys():
        organ_dis_dict[dis_name] = []
    if 'normal' not in organ_dis_dict.keys():
        organ_dis_dict['normal'] = []
    ###---Sentence dictonary Keys.
    if 'normal' not in sentence_dis_dict.keys():
        sentence_dis_dict['normal'] = []
    if dis_name not in sentence_dis_dict.keys():
        sentence_dis_dict[dis_name] = []
    for sent in sentence_list:
        organ_sub = re.search(organ_sub_re, sent)
        organ_try_list.append(organ_sub)
        disease = re.search(dis_re, sent)
        disease_list.append(disease)
        deny = re.search(deny_re, sent)
        # specific disease
        # specific disease
        if is_specific==1 and deny is None and disease is not None and caught==0:
            (organ_dis_dict[dis_name]).append(sub)
            caught = 1
            dis_caught_flag = 1
            update_dict(write_dict, sub, dis_name, disease.group(0))
            (sentence_dis_dict[dis_name]).append(sent)
        # common disease
        elif is_specific==0 and organ_sub is not None and deny is None and disease is not None and caught==0:
            (organ_dis_dict[dis_name]).append(sub)
            caught = 1
            dis_caught_flag = 1
            update_dict(write_dict, sub, dis_name, disease.group(0))
            #print(write_dict)
            update_dict(write_dict, sub, organ_list[0], organ_sub.group(0))
            (sentence_dis_dict[dis_name]).append(sent)
           # print(write_dict)

    return organ_dis_dict, write_dict, dis_caught_flag,sentence_dis_dict


# Searching Normal term
def search_normal(organ_list, report, organ_dis_dict, write_dict,sentence_dis_dict):
    sub = report[0]
    ###Sentence_tokanizer for the Findings and Impressions.
    #a=report[2].lower()
    a=report[2]
    sentence_list = sent_tokenize(a)

    organ_re = r'\s' + organ_list[0] + r'|' + organ_list[0].capitalize()

    organ_adj_re = r''
    for re_tok in organ_list[1]:
        organ_adj_re = organ_adj_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
    organ_adj_re = organ_adj_re[:-1]

    normal_n_re = r'\snormal|Normal|\sunremarkable|Unremarkable|\snegative\s+exam|Negative\s+exam|\spatent|Patent|\sclear|Clear\s'

    normal_adj_re = r'\sno\s(\w|\s|,)*abnormalit|No\s(\w|\s|,)*abnormalit|\swithout\s(\w|\s|,)*abnormalit|Without\s(\w|\s|,)*abnormalit'

    other_normal_re = r'\sother|Other|\sacute|\ssize|size|\scontour|Contour|\sattenuation|Attenuation|\scaliber|Caliber|lungs\s+base|\showever\s|' \
                                 r'Lung\s+base|lungs\s+base|\smorphological|Morphological|\srather\s|Rather\s'
    caught = 0
    if 'normal' not in organ_dis_dict.keys():
        organ_dis_dict['normal'] = []
    for sent in sentence_list:
        normal_n = re.search(normal_n_re, sent)
        normal_adj = re.search(normal_adj_re, sent)
        organ_n = re.search(organ_re, sent)
        organ_adj = re.search(organ_adj_re, sent)
        other_normal = re.search(other_normal_re, sent)
        length_of_sentence=len(sent.split())
        if normal_n is not None and organ_n is not None and caught==0 and other_normal is None and (length_of_sentence<=15):
            (organ_dis_dict['normal']).append(sub)
            caught = 1
            update_dict(write_dict, sub, 'normal', normal_n.group(0))
            update_dict(write_dict, sub, organ_list[0], organ_n.group(0))
            (sentence_dis_dict['normal']).append(sent)
        if normal_adj is not None and organ_adj is not None and caught==0 and other_normal is None and (length_of_sentence<=15):
            (organ_dis_dict['normal']).append(sub)
            caught = 1
            update_dict(write_dict, sub, 'normal', normal_adj.group(0))
            update_dict(write_dict, sub, organ_list[0], organ_adj.group(0))
            (sentence_dis_dict['normal']).append(sent)
    return organ_dis_dict, write_dict, sentence_dis_dict


########################

liver_organ_list = LIVER_ORGAN_LIST


common_dis_list = COMMON_DIS_LIST
liver_dis_list = LIVER_DIS_LIST
abandon_list = ABANDON_LIST

# liver
liver_dict = {}
liver_write_dict = {}
sent_dict={}


# liver
for i in range(len(split_list)):
#for i in range(6,7):

    print('Processing n0====',i)
    dis_caught_flag=0
    # common disease
    for dis in common_dis_list:
        liver_dict, liver_write_dict, dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, is_specific=0, organ_list=liver_organ_list, report=split_list[i], organ_dis_dict=liver_dict, write_dict=liver_write_dict,sentence_dis_dict=sent_dict)
        #specific disease
    for dis in liver_dis_list:
        liver_dict, liver_write_dict, dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, is_specific=1, organ_list=liver_organ_list, report=split_list[i], organ_dis_dict=liver_dict, write_dict=liver_write_dict,sentence_dis_dict=sent_dict)
    if dis_caught_flag==0:
        # normal
        liver_dict, liver_write_dict,sent_dict = search_normal(organ_list=liver_organ_list, report=split_list[i], organ_dis_dict=liver_dict, write_dict=liver_write_dict,sentence_dis_dict=sent_dict)
    else:
        continue


liver_normal_list = []
for sub in liver_dict['normal']:
    #if sub not in liver_abnormal_list:
        liver_normal_list.append(sub)

Inf0_normal=pd.DataFrame(list(zip(liver_normal_list)),
columns=['id'])
Inf0_normal.to_csv(PATH_TO_SAVE_CSV+"liver_normal.csv", encoding='utf-8', index=False)

liver_abnormal_list = []
for i in range(len(split_list)):
    sub = split_list[i][0]
    for dis in common_dis_list:
        if sub in liver_dict[dis] and sub not in liver_abnormal_list:
             liver_abnormal_list.append(sub)
    for dis in liver_dis_list:
        if sub in liver_dict[dis] and sub not in liver_abnormal_list:
             liver_abnormal_list.append(sub)


Inf0_abnormal=pd.DataFrame(list(zip(liver_abnormal_list)),
columns=['id'])
Inf0_abnormal.to_csv(PATH_TO_SAVE_CSV+"liver_abnormal.csv", encoding='utf-8', index=False)




liver_unknown_list = []
for i in range(len(split_list)):
    sub = split_list[i][0]
    if sub not in liver_abnormal_list and sub not in liver_normal_list:
        liver_unknown_list.append(sub)

Inf0_unknown=pd.DataFrame(list(zip(liver_unknown_list)),
columns=['id'])
Inf0_unknown.to_csv(PATH_TO_SAVE_CSV+"liver_unknown.csv", encoding='utf-8', index=False)

####################### Saving Dictonary###################
liver_dict_pickle_out = open(PATH_TO_SAVE_CSV+"liver_dict.pickle","wb")
pickle.dump(liver_dict, liver_dict_pickle_out)
liver_dict_pickle_out.close()

liver_write_dict_pickle_out = open(PATH_TO_SAVE_CSV+"liver_write_dict.pickle","wb")
pickle.dump(liver_write_dict, liver_write_dict_pickle_out)
liver_write_dict_pickle_out.close()

sent_dict_out = open(PATH_TO_SAVE_CSV+"Liver_sentence_dict.pickle","wb")
pickle.dump(sent_dict, sent_dict_out)
sent_dict_out.close()

#################### Saving Directory #####################

#############----- Making the Sentence List------##############

def make_Sentence_list_with_length(key_name,Dictonary,csv_name):
    sentence_list = []
    ##Getting The sentence
    for sub in Dictonary[key_name]:
        sentence_list.append(sub)
    sentence_length=[]
    for i in range(len(sentence_list)):
        sent=str(sentence_list[i])
        length_of_sentence=len(sent.split())
        sentence_length.append(length_of_sentence)

    csv_save=pd.DataFrame(list(zip(sentence_list,sentence_length)),columns=['Sentence','length'])
    csv_save.to_csv(csv_name,encoding='utf-8', index=False)
    return csv_save


make_Sentence_list_with_length('normal',sent_dict,PATH_TO_SAVE_CSV+'Liver_Normal_Sentence.csv')
##---Liver Stone Class
Stone_Sentence_csv=make_Sentence_list_with_length('stone',sent_dict,PATH_TO_SAVE_CSV+'Liver_Stone_Sentence.csv')
Gallstone_Sentence_csv=make_Sentence_list_with_length('gallstone',sent_dict,PATH_TO_SAVE_CSV+'Liver_Gallstone_Sentence.csv')
Calcifi_Sentence_csv=make_Sentence_list_with_length('calcifi',sent_dict,PATH_TO_SAVE_CSV+'Liver_Calcifi_Sentence.csv')
Liver_Train_StoneClass_Sentence_March16_2020=pd.concat([Stone_Sentence_csv,Gallstone_Sentence_csv,Calcifi_Sentence_csv])
Liver_Train_StoneClass_Sentence_March16_2020.to_csv(PATH_TO_SAVE_CSV+"Liver_StoneClass_Sentence.csv", encoding='utf-8', index=False)
###----Lesion Class
Lesion_Sentence_csv=make_Sentence_list_with_length('lesion',sent_dict,PATH_TO_SAVE_CSV+'Liver_Lesion_Sentence.csv')
Mass_Sentence_csv=make_Sentence_list_with_length('mass',sent_dict,PATH_TO_SAVE_CSV+'Liver_Mass_Sentence.csv')
Liver_Train_LesionClass_Sentence_March16_2020=pd.concat([Lesion_Sentence_csv,Mass_Sentence_csv])
Liver_Train_LesionClass_Sentence_March16_2020.to_csv(PATH_TO_SAVE_CSV+"Liver_LesionClass_Sentence.csv", encoding='utf-8', index=False)
###--Dilat Class
make_Sentence_list_with_length('dilat',sent_dict,PATH_TO_SAVE_CSV+'Liver_DilatClass_Sentence.csv')
###----Lesion Class
Fatty_Sentence_csv=make_Sentence_list_with_length('fatty\s+infiltr',sent_dict,PATH_TO_SAVE_CSV+'Liver_Fatty_Sentence.csv')
Steatosis_Sentence_csv=make_Sentence_list_with_length('steatosis',sent_dict,PATH_TO_SAVE_CSV+'Liver_steatosis_Sentence.csv')
Liver_Train_FattyClass_Sentence_March16_2020=pd.concat([Fatty_Sentence_csv,Steatosis_Sentence_csv])
Liver_Train_FattyClass_Sentence_March16_2020.to_csv(PATH_TO_SAVE_CSV+"Liver_FattyClass_Sentence.csv", encoding='utf-8', index=False)


###############--for Sentence Csv Making--------###################

Sentence_key_list = [ 'mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren',  'hyperenhanc' , 'hypoenhanc','aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct', 'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural distortion', 'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi', 'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent', 'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent','steatosis', 'cirrho','cholecystectomy','gallstone','cholelithiasis']

for dis in Sentence_key_list:
    print('Disease-->>{}--- Sentence Csv'.format(dis))
    if dis=='surgically\s+absent':
        name= PATH_TO_SAVE_CSV+'Liver_surgically_absent_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='fatty\\s+infiltr':
        name= PATH_TO_SAVE_CSV+'Liver_fatty_infiltr_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)
    else:
        name=PATH_TO_SAVE_CSV+'Liver_'+dis+'_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

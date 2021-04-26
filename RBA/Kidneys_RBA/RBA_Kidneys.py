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
from RBA_Kidneys_Config import*

####################Reading the Report CSV###################
## Reading the Report CSV
mylist = pd.read_csv(
        REPORT_CSV,
        dtype=object,
        keep_default_na=False,
        na_values=[]).as_matrix()
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
#print("Looping through the report_list of the length: ",len(report_list))
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
    #print("Split list was appended: ",split_list)

############################# Functions #############################################
##Upading the Dictonary
def update_dict(thedict, key_a, key_b, val):
    adic = thedict.keys()
    if key_a in adic:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

## Diseased Term
kidn=[]
kidney_dis=[]
kid_den=[]
def search_dis(dis_caught_flag, dis_name, organ_list, report, organ_dis_dict, write_dict,sentence_dis_dict,is_specific=0):
    ##Subject Id
    sub = report[0]

    ###Adding the Finding the Impression
    #a=report[2].lower()
    a=report[2]
    sentence_list = sent_tokenize(a)
    organ_sub_re = r''
    for re_tok in organ_list[2]:
        #Lower and uppercase options and making kidney as wild card TRied to make renal wild card but that introduce additional drop in performance.
        if re_tok=='kidney':
            organ_sub_re = organ_sub_re + '\s*' + re_tok+'*'+ r'|\s*' + re_tok.capitalize()+'*'+ r'|'
        else:
            organ_sub_re = organ_sub_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
    organ_sub_re = organ_sub_re[:-1]
    #adding lower/uppercase disease names
    dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()

   ## Extra layer of typo protection
    No_with_organ_Descriptor=r''
    for name_of_organ_descriptor in organ_list[2]:
        No_with_organ_Descriptor=No_with_organ_Descriptor+ '\s*' +'no'+name_of_organ_descriptor+'*'+ r'|\s*' +'No'+name_of_organ_descriptor+'*'+ r'|'
    No_with_organ_Descriptor=No_with_organ_Descriptor[:-1]


    #adding lower/uppercase denial words
    deny_re = r'\sno\s|No\s|no\s|\snon|Non\s|\sother|Other\s|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
              r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
              r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter'

    if dis_name == 'stone':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*|\s*nonobstructing\s+stone*|\s*Nonobstructing\s+stone*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non\s|\sNo|No|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
        r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
        r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\sphlebolith\s|Phlebolith\s|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter'


    if dis_name == 'calcifi':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'
        deny_re =r'\sno\s|No\s|no\s|\snon|Non\s|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
                 r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*lesion*|\s*Lesion*|' \
                 r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\sphlebolith\s|Phlebolith\s|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter'


    if dis_name=='calcul':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*|\s*nonobstructing\s+calculus*|\s*Nonobstructing\s+calculus*'
        deny_re = r'\sno\s|No\s|no\s|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*calculation*|\s*Calculation*|\s*lesion*|\s*Lesion*|' \
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter'


    if dis_name=='lesion':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'
        deny_re = r'\sno\s|No\s|no\s|Nofocal\s|\sNofocal\s|\snofocal\s|nofocal\s|\snon\s|\snon|Non\s|\snot\s|Not\s|\snone|None\s|\s*without*|\s*Without*|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|Nohydronephrosis\s|\snohydronephrosis\s|nohydronephrosis\s|Nohydroureteronephrosis\s|\snohydroureteronephrosis\s|nohydroureteronephrosis\s|' \
                  r'\s*like*(\w|\s|,)*cyst|\s*Like*(\w|\s|,)*cyst|\s*consistent*(\w|\s|,)*cyst|\s*consistent*(\w|\s|,)*cyst|\s*represent*(\w|\s|,)*cyst|\s*Represent*(\w|\s|,)*cyst|\s*compatible*(\w|\s|,)*cyst|\s*Compatible*(\w|\s|,)*cyst|'\
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter|\s*adrenal*|\s*Adrenal*|'\
                  r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'+'|'+ No_with_organ_Descriptor
        organ_sub_re = r''
        for re_tok in organ_list[2]:
        #Lower and uppercase options
            if re_tok=='kidney':
                organ_sub_re = organ_sub_re + '\s*' + re_tok+'*'+ r'|\s*' + re_tok.capitalize()+'*'+ r'|'
            if re_tok=='renal':
                organ_sub_re = organ_sub_re + '\s*' + re_tok+'*'+ r'|\s*' + re_tok.capitalize()+'*'+ r'|'
            else:
                organ_sub_re = organ_sub_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
        organ_sub_re = organ_sub_re[:-1]

    if dis_name=='cyst':
        dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()+'|\scysts|Cysts\s|\scysts\s|\sCysts\s|\scystic\s|Cystic\s'+'|\s*nonenhancing\s+cyst*|\s*Nonenhancing\s+cyst*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non\s|\sother|Other\s|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*adrenal*|\s*Adrenal*|\s*cystectomy*|\s*Cystectomy*|\scystic\s+space\s|' \
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter|\s*adrenal*|\s*Adrenal*'
        organ_sub_re = r''
        for re_tok in organ_list[2]:
        #Lower and uppercase options
            if re_tok=='kidney':
                organ_sub_re = organ_sub_re + '\s*' + re_tok+'*'+ r'|\s*' + re_tok.capitalize()+'*'+ r'|'
            if re_tok=='renal':
                organ_sub_re = organ_sub_re + '\s*' + re_tok+'*'+ r'|\s*' + re_tok.capitalize()+'*'+ r'|'
            else:
                organ_sub_re = organ_sub_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
        organ_sub_re = organ_sub_re[:-1]

    ### Adding wild cards for Surgically Absent
    if dis_name=='surgically\s+absent':
        dis_re=r'\s*surgically\s+absent*|\s*Surgically\s+absent*|\s*surgicallyabsent*|\s*Surgicallyabsent*|\s*surgically\s+removed*|\s*Surgically\s+removed*|\s*surgicallyremoved*|\s*Surgicallyremoved*|\s*absent*|\s*Absent*'
    ##---Fatty wild card
    if dis_name=='fatty\s+infiltr':
        dis_re=r'\s*fatty\s+infiltr*|\s*Fatty\s+infiltr*|\s*fattyinfiltr*|\s*Fattyinfiltr*'

    if dis_name == 'mass':
        dis_re=r'\s*mass*|\s*Mass*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\sisno\s|\sareno\s|Noevidence\s|\snoevidence\s|Nosuspicious\s|\snosuspicious\s|\snosuspicious|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\smassive|Massive|' \
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|\smass\s+effect|Mass\s+effect||\srenal\s+arter|Renal\s+arter|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter|'\
                  r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'+'|'+ No_with_organ_Descriptor
    caught = 0
    #if the name of the disease that we provided to the function is not already
    #in the organ disease list create a key with this disease name
    if 'normal' not in organ_dis_dict.keys():
        #creating 'normal' key in the dictionary
        organ_dis_dict['normal'] = []
    if dis_name not in organ_dis_dict.keys():
        organ_dis_dict[dis_name] = []

    ###---Sentence dictonary Keys.
    if 'normal' not in sentence_dis_dict.keys():
        sentence_dis_dict['normal'] = []
    if dis_name not in sentence_dis_dict.keys():
        sentence_dis_dict[dis_name] = []

    for sent in sentence_list:
        #For each word in the sentence
        #Check if the word corresponds to the organ name
        organ_sub = re.search(organ_sub_re, sent)
        kidn.append(organ_sub)
        #Check if the word corresponds to a disease name we are checking in this function call
        disease = re.search(dis_re, sent)
        kidney_dis.append(disease)
        #Check if the word corresponds to the deny word
        deny = re.search(deny_re, sent)
        kid_den.append(deny)
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
            update_dict(write_dict, sub, organ_list[0], organ_sub.group(0))
            (sentence_dis_dict[dis_name]).append(sent)

    return organ_dis_dict, write_dict, dis_caught_flag, sentence_dis_dict

# Searching Normal term
def search_normal(organ_list, report, organ_dis_dict, write_dict,sentence_dis_dict):
    ### Subject Id
    sub = report[0]
    ###Sentence_tokanizer for the Findings and Impressions.
    ###report lowercase
    #a=report[2].lower()
    a=report[2]
    sentence_list = sent_tokenize(a)
    #Mention of organ lower/uppercase
    organ_re = r'\s' + organ_list[0] + r'|' + organ_list[0].capitalize()

    organ_adj_re = r''
    for re_tok in organ_list[1]:
        organ_adj_re = organ_adj_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
    organ_adj_re = organ_adj_re[:-1]

    normal_n_re = r'\snormal|Normal|\snremarkable|Unremarkable|\snegative\s+exam|Negative\s+exam|\spatent|Patent|\sclear\s|Clear\s'

    normal_adj_re = r'\sno\s(\w|\s|,)*abnormalit|No\s(\w|\s|,)*abnormalit|\swithout\s(\w|\s|,)*abnormalit|Without\s(\w|\s|,)*abnormalit'

    other_normal_re = r'\sother|Other|\snew|New|\sacute|\ssize|Size|\scontour|Contour|\sattenuation|Attenuation|\scaliber|Caliber|lungs\s+base|\srather\s|Rather\s|\showever\s|' \
                                 r'lung\s+base|Lung\s+base|Lungs\s+base|\smorphological|Morphological|renal\s+arteries|\srenal\s+artery|Renal\s+arteries|Renal\s+artery|\srenal\s(\w|\s|,)*arter|Renal\s(\w|\s|,)*arter'
    caught = 0
    if 'normal' not in organ_dis_dict.keys():
        #creating 'normal' key in the dictionary
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

    return organ_dis_dict, write_dict,sentence_dis_dict


############################### Sorted List #######################

########################

kidney_organ_list = KIDNEYS_ORGAN_LIST
common_dis_list = COMMON_DIS_LIST
kidney_dis_list = KIDNEYS_DIS_LIST



# Kidney
kidney_dict = {}
kidney_write_dict = {}
sent_dict={}
# kidney
for i in range(len(split_list)):
#for i in range(20,21):
    print('Processing n0====',i)
    # common disease
    dis_caught_flag=0
    for dis in common_dis_list:
        kidney_dict, kidney_write_dict,dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, organ_list=kidney_organ_list, report=split_list[i], organ_dis_dict=kidney_dict, write_dict=kidney_write_dict, is_specific=0,sentence_dis_dict=sent_dict)
    #specific disease
    for dis in kidney_dis_list:
        kidney_dict, kidney_write_dict, dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, is_specific=1, organ_list=kidney_organ_list, report=split_list[i], organ_dis_dict=kidney_dict, write_dict=kidney_write_dict,sentence_dis_dict=sent_dict)
    if dis_caught_flag==0:
        kidney_dict, kidney_write_dict,sent_dict = search_normal(organ_list=kidney_organ_list, report=split_list[i], organ_dis_dict=kidney_dict, write_dict=kidney_write_dict,sentence_dis_dict=sent_dict)
    else:
        continue



kidney_normal_list = []
for sub in kidney_dict['normal']:
    #if sub not in kidney_abnormal_list:
        kidney_normal_list.append(sub)
Inf0_normal=pd.DataFrame(list(zip(kidney_normal_list)),
columns=['id'])
Inf0_normal.to_csv("Kidney_normal.csv", encoding='utf-8', index=False)

kidney_abnormal_list = []
for i in range(len(split_list)):
    sub = split_list[i][0]
    for dis in common_dis_list:
        if sub in kidney_dict[dis] and sub not in kidney_abnormal_list:
             kidney_abnormal_list.append(sub)

Inf0_abnormal=pd.DataFrame(list(zip(kidney_abnormal_list)),
columns=['id'])
Inf0_abnormal.to_csv("Kidney_abnormal.csv", encoding='utf-8', index=False)


kidney_unknown_list = []
for i in range(len(split_list)):
    sub = split_list[i][0]
    if sub not in kidney_abnormal_list and sub not in kidney_normal_list:
        kidney_unknown_list.append(sub)

Inf0_unknown=pd.DataFrame(list(zip(kidney_unknown_list)),
columns=['id'])
Inf0_unknown.to_csv("Kidney_unknown.csv", encoding='utf-8', index=False)


####################### Saving Dictonary###################
kidney_dict_pickle_out = open("kidney_dict.pickle","wb")
pickle.dump(kidney_dict, kidney_dict_pickle_out)
kidney_dict_pickle_out.close()

kidney_write_dict_pickle_out = open("kidney_write_dict.pickle","wb")
pickle.dump(kidney_write_dict, kidney_write_dict_pickle_out)
kidney_write_dict_pickle_out.close()


sent_dict_out = open("kidney_Sentence_dict.pickle","wb")
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

#############################Sentence---csv-------------##########################
make_Sentence_list_with_length('normal',sent_dict,'Kidney_NormalClass_Sentence.csv')
###----Stone Class-----------
Stone_Sentence=make_Sentence_list_with_length('stone',sent_dict,'kidney_Stone_sentence.csv')
calcul_Sentence=make_Sentence_list_with_length('calcul',sent_dict,'kidney_Stone_sentence.csv')
calcifi_Sentence=make_Sentence_list_with_length('calcifi',sent_dict,'kidney_Stone_sentence.csv')
Kidney_StoneClass_Sentence=pd.concat([Stone_Sentence,calcul_Sentence,calcifi_Sentence])
Kidney_StoneClass_Sentence.to_csv('Kidney_StoneClass_Sentence.csv',encoding='utf-8',index=False)
###----Atrophy class-------
Atrophy_Sentence=make_Sentence_list_with_length('atrophy',sent_dict,'kidney_Atrophy_sentence.csv')
Atroph_Sentence=make_Sentence_list_with_length('atroph',sent_dict,'kidney_Atroph_sentence.csv')
Kidney_AtrophClass_Sentence=pd.concat([Atroph_Sentence,Atrophy_Sentence])
Kidney_AtrophClass_Sentence.to_csv('Kidney_AtrophClass_Sentence.csv',encoding='utf-8',index=False)

Kidney_LesionClass_Sentence=make_Sentence_list_with_length('lesion',sent_dict,'Kidney_LesionClass_Sentence.csv')
Kidney_CystClass_Sentence=make_Sentence_list_with_length('cyst',sent_dict,'Kidney_CystClass_Sentence.csv')



Sentence_key_list = ['mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren', 'hyperenhanc' , 'hypoenhanc', 'aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct', 'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural distortion', 'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi', 'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent', 'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent', 'hydronephrosis', 'hydroureter', 'nephrectomy', 'percutaneous', 'pelvicaliectasis', 'uropathy', 'ureterectasis','nephrolithiasis']


for dis in Sentence_key_list:
    print('Disease-->>{}--- Sentence Csv'.format(dis))
    if dis=='surgically\s+absent':
        name= 'Kidney_surgically_absent_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='fatty\\s+infiltr':
        name= 'Kidney_fatty_infiltr_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)
    else:
        name='Kidney_'+dis+'_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

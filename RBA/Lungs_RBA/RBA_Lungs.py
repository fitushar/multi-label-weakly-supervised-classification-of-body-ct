"""
@author: Fakrul Islam Tushar (fakrulislam.tushar@duke.edu)

© copyright 2020 - 2021 Duke University
-This program is free software and a property of Duke University:
you can redistribute it and/or modify for non-commercial uses only.
-This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

"""
###############


###################import libraries#########################

import pandas as pd
import numpy as np
import glob
import os
import re
import csv
import pickle
import seaborn as sn
from RBA_Lung_Config import*
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize

####################Reading the Report CSV###################
## Reading the Report CSV
mylist = pd.read_csv(REPORT_CSV,dtype=object,keep_default_na=False,na_values=[])
#mylist=mylist.drop_duplicates(subset='Report Text', keep='first')
mylist = mylist.as_matrix()
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

############################# Functions #############################################
##Upading the Dictonary
def update_dict(thedict, key_a, key_b, val):
    adic = thedict.keys()
    if key_a in adic:
        thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})

## Diseased Term
def search_dis(dis_caught_flag, dis_name, organ_list, report, organ_dis_dict, write_dict, sentence_dis_dict,is_specific=0):
    ##Subject Id
    sub = report[0]

    ###Adding to Finding the Impression for lung
    #a=report[2].lower()
    #if dis_name=='pneumoni':
    #    a=report[2]
    #    a=a+' '+report[3]
    #else:
    a=report[2]
    sentence_list = sent_tokenize(a)

    organ_sub_re = r''
    for re_tok in organ_list[2]:
        organ_sub_re = organ_sub_re + '\s' + re_tok + r'|' + re_tok.capitalize() + r'|'
    organ_sub_re = organ_sub_re[:-1]

    #dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()
    dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'  #--| Using the disease names as wildCards
    #dis_re= r'\s*' + dis_name+'*|\s*' + dis_name.capitalize()+'*'  #--| Using the disease names as wildCards


    deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|'\
    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    if dis_name=='edema':
        dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()

    if dis_name=='stent':
        dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()

    if dis_name=='ectasia':
        dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()

    if dis_name=='ground\s+glass':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'+ r'|\s*groundglass*|\s*Groundglass*'

    #---Emphysema wild-Card.
    if dis_name=='emphysema':
        dis_re=r'\s*emphysema*|\s*Emphysema*'

    #---|Lesion
    if dis_name=='lesion':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'

    ###--|Ground Glass
    if dis_name=='ground\s+glass':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'+ r'|\s*groundglass*|\s*Groundglass*'
    ###--|Surgically Absent
    if dis_name=='surgically\s+absent':
        dis_re=r'\s*surgically\s+absent*|\s*Surgically\s+absent*|\s*surgicallyabsent*|\s*Surgicallyabsent*|\s*surgically\s+removed*|\s*Surgically\s+removed*|\s*surgicallyremoved*|\s*Surgicallyremoved*|\s*absent*|\s*Absent*'
    ##---|Fatty wild card
    if dis_name=='fatty\s+infiltr':
        dis_re=r'\s*fatty\s+infiltr*|\s*Fatty\s+infiltr*|\s*fattyinfiltr*|\s*Fattyinfiltr*'

    ###--|Ground Glass
    if dis_name=='air\s+trapping':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'+ r'|\s*airtrapping*|\s*Airtrapping*'

    ###--|Architectural distortion
    if dis_name=='architectural\s+distortion':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'+ r'|\s*architecturaldistortion*|\s*Architecturaldistortion*'

    if dis_name=='scar':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*scattered*|\s*Scattered*|' \
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|'\
                  r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    if dis_name=='calcul':
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*calculation*|\s*Calculation*|' \
                r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|'\
                r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'


    ##--Atelecta
    if dis_name == 'atelecta':
        dis_re=r'\s*atelecta*|\s*Atelecta*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
                    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|dependent\s|Dependent\s|'\
                    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    ##--|Pneumoni
    if dis_name == 'pneumoni':
        dis_re=dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\s*pneumonectomy*|\s*Pneumonectomy*|'\
                    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|dependent\s|Dependent\s|'\
                    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    #--Nodule
    if dis_name == 'nodule':
        dis_re = r'\s' + dis_name + r'|' + dis_name.capitalize()+r'|'+r'\s' + 'nodules' + r'|' + 'Nodules'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
                    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|\scalcifi|Calcifi|'\
                    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    #---|Mass
    if dis_name == 'mass':
        dis_re=r'\s*mass*|\s*Mass*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|' \
                    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|\srather\s|Rather\s|\smass\s+effect|Mass\s+effect|'\
                    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    #---|Effusion
    if dis_name=='effusion':
        dis_re= r'\s*' + dis_name+'*' + r'|\s*' + dis_name.capitalize()+'*'
        #organ_sub_re=organ_sub_re + '|\s*pleural*|\s*Pleural*'
        deny_re = r'\sno\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\s*nopleural*|\s*Nopleural*|' \
                    r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                    r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|'\
                    r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'

    #---|Pleural effusion
    if dis_name=='pleural\s+effusion':
        dis_re= r'\s*pleural\s+effusion*|\s*Pleural\s+effusion*|\s*pleuraleffusion*|\s*pleuraleffusion*'
        deny_re = r'\sno\s|No\s|no\s|No\s|no\s|\snon|Non|\sother|Other|\snot\s|Not\s|\snone|None\s|\swithout\s|Without\s|\srather\s|Rather\s|\s*nopleural*|\s*Nopleural*|' \
                  r'\snegative\s|Negative\s|with\s+regards\s+to|With\s+regards\s+to|\showever\s|\sisno\s|\sareno\s|Noevidence|\snoevidence\s|' \
                  r'\slimited\s+exam\s+for\s+the\s+evalution|Limited\s+exam\s+for\s+the\s+evalution|'\
                  r'\s*' + 'no'+dis_name+'*' + r'|\s*No'+dis_name+'*|'+r'\s*' + 'not'+dis_name+'*' + r'|\s*Not'+dis_name+'*'


    caught = 0
    if 'normal' not in organ_dis_dict.keys():
        organ_dis_dict['normal'] = []
    if dis_name not in organ_dis_dict.keys():
        organ_dis_dict[dis_name] = []

    if 'normal' not in sentence_dis_dict.keys():
        sentence_dis_dict['normal'] = []
    if dis_name not in sentence_dis_dict.keys():
        sentence_dis_dict[dis_name] = []
    for sent in sentence_list:
        organ_sub = re.search(organ_sub_re, sent)

        disease = re.search(dis_re, sent)
        deny = re.search(deny_re, sent)

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
unk_lung_list=[]
# Searching Normal term
def search_normal(organ_list, report, organ_dis_dict, write_dict,sentence_dis_dict):
    ### Subject Id
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

    normal_n_re = r'\snormal|Normal|\sunremarkable|Unremarkable|\snegative\s+exam|Negative\s+exam|\spatent|Patent|\sclear|Clear\s|' \
    r'\spatent|patent\s'
    normal_adj_re = r'\sno\s(\w|\s|,)*abnormalit|No\s(\w|\s|,)*abnormalit|\swithout\s(\w|\s|,)*abnormalit|Without\s(\w|\s|,)*abnormalit'

    other_normal_re = r'\sother|Other|Acute|\sacute|\snew|New|\ssize|Size|\scontour|Contour|\sattenuation|Attenuation|\scaliber|Caliber|\srather\s|Rather\s|\showever\s|' \
                                 r'\smorphological|Morphological'
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

        if normal_n is not None and organ_n is not None and caught==0 and other_normal is None and (length_of_sentence<=10):
            (organ_dis_dict['normal']).append(sub)
            caught = 1
            update_dict(write_dict, sub, 'normal', normal_n.group(0))
            update_dict(write_dict, sub, organ_list[0], organ_n.group(0))
            (sentence_dis_dict['normal']).append(sent)
        if normal_adj is not None and organ_adj is not None and caught==0 and other_normal is None and (length_of_sentence<=10):
            (organ_dis_dict['normal']).append(sub)
            caught = 1
            update_dict(write_dict, sub, 'normal', normal_adj.group(0))
            update_dict(write_dict, sub, organ_list[0], organ_adj.group(0))
            (sentence_dis_dict['normal']).append(sent)


    return organ_dis_dict, write_dict,sentence_dis_dict



####---| Getting the input
lung_organ_list = LUNG_ORGAN_LIST
common_dis_list = COMMON_DIS_LIST
lung_dis_list   = LUNG_DIS_LIST


#-Creating the Dictonaries lung
lung_dict       = {}
lung_write_dict = {}
sent_dict       = {}


# lung
for i in range(len(split_list)):
#for i in range(0,52):
    print('Processing n0====',i)
    dis_caught_flag = 0
    # common disease
    for dis in common_dis_list:
        lung_dict, lung_write_dict, dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, is_specific=0, organ_list=lung_organ_list, report=split_list[i], organ_dis_dict=lung_dict, write_dict=lung_write_dict,sentence_dis_dict=sent_dict)
    #specific disease
    for dis in lung_dis_list:
        lung_dict, lung_write_dict, dis_caught_flag,sent_dict = search_dis(dis_caught_flag, dis_name=dis, is_specific=1, organ_list=lung_organ_list, report=split_list[i], organ_dis_dict=lung_dict, write_dict=lung_write_dict,sentence_dis_dict=sent_dict)

    if dis_caught_flag==0:
        # normal
        lung_dict, lung_write_dict,sent_dict = search_normal(organ_list=lung_organ_list, report=split_list[i], organ_dis_dict=lung_dict, write_dict=lung_write_dict,sentence_dis_dict=sent_dict)
    else:
        continue


####-----| Saving Lung Normal, Abnormal, and unknown CSV
lung_normal_list = []
lung_abnormal_list = []
lung_unknown_list = []

#---| Normal CSV
for sub in lung_dict['normal']:
    #if sub not in lung_abnormal_list:
        lung_normal_list.append(sub)
Inf0_normal=pd.DataFrame(list(zip(lung_normal_list)),columns=['id'])
Inf0_normal.to_csv(PATH_TO_SAVE_CSV+"lung_normal.csv", encoding='utf-8', index=False)

#----|Abnormal CSV
for i in range(len(split_list)):
    sub = split_list[i][0]
    for dis in common_dis_list:
        if sub in lung_dict[dis] and sub not in lung_abnormal_list:
             lung_abnormal_list.append(sub)
    for dis in lung_dis_list:
        if sub in lung_dict[dis] and sub not in lung_abnormal_list:
             lung_abnormal_list.append(sub)
Inf0_abnormal=pd.DataFrame(list(zip(lung_abnormal_list)),columns=['id'])
Inf0_abnormal.to_csv(PATH_TO_SAVE_CSV+"lung_abnormal.csv", encoding='utf-8', index=False)

#----|Unknown CSV
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))
for i in range(len(split_list)):
    sub = split_list[i][0]
    if sub not in lung_abnormal_list and sub not in lung_normal_list:
        lung_unknown_list.append(sub)
Inf0_unknown=pd.DataFrame(list(zip(lung_unknown_list)),columns=['id'])
Inf0_unknown.to_csv(PATH_TO_SAVE_CSV+"lung_unknown.csv", encoding='utf-8', index=False)


####################### Saving Dictonary###################
lung_dict_pickle_out = open(PATH_TO_SAVE_CSV+"lung_dict.pickle","wb")
pickle.dump(lung_dict, lung_dict_pickle_out)
lung_dict_pickle_out.close()

sent_dict_out = open(PATH_TO_SAVE_CSV+"lung_sentence_dict.pickle","wb")
pickle.dump(sent_dict, sent_dict_out)
sent_dict_out.close()

lung_write_dict_pickle_out = open(PATH_TO_SAVE_CSV+"lung_write_dict.pickle","wb")
pickle.dump(lung_write_dict, lung_write_dict_pickle_out)
lung_write_dict_pickle_out.close()


############################-----Making the sentence list---###############################
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

Lung_NormalClass_Sentence=make_Sentence_list_with_length('normal',sent_dict,PATH_TO_SAVE_CSV+'Lung_NormalClass_Sentence.csv')
##--- Lung Atelectasis Class
Lung_AtelectaClass_Sentence=make_Sentence_list_with_length('atelecta',sent_dict,PATH_TO_SAVE_CSV+'Lung_AtelectaClass_Sentence.csv')
Lung_PneumoniClass_Sentence=make_Sentence_list_with_length('pneumoni',sent_dict,PATH_TO_SAVE_CSV+'Lung_PneumoniClass_Sentence.csv')
Lung_AtelectasisClass_Sentence=pd.concat([Lung_AtelectaClass_Sentence,Lung_PneumoniClass_Sentence])
Lung_AtelectasisClass_Sentence.to_csv(PATH_TO_SAVE_CSV+"Lung_NoduleClass_Sentence.csv", encoding='utf-8', index=False)
##---Lung Nodule Class
Lung_NodularClass_Sentence=make_Sentence_list_with_length('nodular',sent_dict,PATH_TO_SAVE_CSV+'Lung_NodularClass_Sentence.csv')
Lung_noduleClass_Sentence=make_Sentence_list_with_length('nodule',sent_dict,PATH_TO_SAVE_CSV+'Lung_noduleClass_Sentence.csv')
Lung_MassClass_Sentence=make_Sentence_list_with_length('mass',sent_dict,PATH_TO_SAVE_CSV+'Lung_MassClass_Sentence.csv')
Lung_NoduleClass_Sentence=pd.concat([Lung_NodularClass_Sentence,Lung_noduleClass_Sentence,Lung_MassClass_Sentence])
Lung_NoduleClass_Sentence.to_csv(PATH_TO_SAVE_CSV+"Lung_NoduleClass_Sentence.csv", encoding='utf-8', index=False)
##--Lung Emphysema Class
Lung_EmphysemaClass_Sentence=make_Sentence_list_with_length('emphysema',sent_dict,PATH_TO_SAVE_CSV+'Lung_EmphysemaClass_Sentence.csv')
##--Lung Effusion Class
Lung_Effusion_Sentence=make_Sentence_list_with_length('effusion',sent_dict,PATH_TO_SAVE_CSV+'Lung_Effusion_Sentence.csv')
Lung_PleuralEffusion_Sentence=make_Sentence_list_with_length('pleural\s+effusion',sent_dict,PATH_TO_SAVE_CSV+'Lung_PleuralEffucion_Sentence.csv')
Lung_EffusioneClass_Sentence=pd.concat([Lung_Effusion_Sentence,Lung_PleuralEffusion_Sentence])
Lung_EffusioneClass_Sentence.to_csv(PATH_TO_SAVE_CSV+"Lung_EffusionClass_Sentence.csv", encoding='utf-8', index=False)


Sentence_key_list = LIST_FOR_OVERLAP_STATISTICS
for dis in Sentence_key_list:
    print('Disease-->>{}--- Sentence Csv'.format(dis))
    if dis=='surgically\s+absent':
        name= PATH_TO_SAVE_CSV+'Lung_surgically_absent_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='fatty\\s+infiltr':
        name= PATH_TO_SAVE_CSV+'Lung_fatty_infiltr_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='ground\\s+glass':
        name= PATH_TO_SAVE_CSV+'Lung_ground_glass_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='air\s+trapping':
        name= PATH_TO_SAVE_CSV+'Lung_air_trapping_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='pleural\s+effusion':
        name= PATH_TO_SAVE_CSV+'Lung_pleural_effusion_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    elif dis=='architectural\s+distortion':
        name= PATH_TO_SAVE_CSV+'Lung_architectural_distortion_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)

    else:
        name=PATH_TO_SAVE_CSV+'Lung_'+dis+'_Sentence.csv'
        make_Sentence_list_with_length(dis,sent_dict,name)
###############################-------------- Statistics------------------------######################

organ_dictonary=PATH_TO_SAVE_CSV+"lung_dict.pickle"
Diagnosis_list = LIST_FOR_OVERLAP_STATISTICS
Disease_Name_and_Number_Decending_CSV_NAME=DISEASE_NAME_AND_NUMBERS
Disease_count_threshold=DISEASE_COUNT_THRESHOLD

###---|Listing Name and Numbers
pickle_in = open(organ_dictonary,"rb")
lung_dict = pickle.load(pickle_in)
for dis in Diagnosis_list:
    cases = lung_dict[dis]
    print('{}--#{}'.format(dis,len(cases)))


name_list=[]
number_list=[]
for i in range(0,len(Diagnosis_list)):
    name=Diagnosis_list[i]
    name_list.append(name)
    cases = lung_dict[name]
    number_list.append(len(cases))

### Saving the Sorted Excel file
DataDisease_data=pd.DataFrame(list(zip(name_list,number_list)),columns=['Disease_name','Number_of_cases'])
Disease_Name_and_Number_Decending= DataDisease_data.sort_values(by ='Number_of_cases',ascending=False)
Disease_Name_and_Number_Decending.to_csv(Disease_Name_and_Number_Decending_CSV_NAME, encoding='utf-8', index=False)
Disease_Name_and_Number_Decending_CSV= pd.read_csv(Disease_Name_and_Number_Decending_CSV_NAME,dtype=object,keep_default_na=False,na_values=[]).as_matrix()


###Comparing with the threshold and give the disease numbers.
Greater_than_Threshold_disease=[]
print("-----------------@@@@@@@@@@------------------------")
print("-----------------Top Diseases----------------------")
print("-----------------@@@@@@@@@@------------------------")
for i in range (0, len(Disease_Name_and_Number_Decending_CSV)):
    Disease_number=int(Disease_Name_and_Number_Decending_CSV[i][1])
    if (Disease_number >= Disease_count_threshold):
        print(Disease_Name_and_Number_Decending_CSV[i][0],Disease_Name_and_Number_Decending_CSV[i][1])
        Greater_than_Threshold_disease.append(Disease_Name_and_Number_Decending_CSV[i][0])

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


for dis in Greater_than_Threshold_disease:
    if dis=='surgically\s+absent':
        name= PATH_TO_SAVE_CSV+'Lung_surgically_absent.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    elif dis=='fatty\\s+infiltr':
        name= PATH_TO_SAVE_CSV+'Lung_fatty_infiltr.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    elif dis=='ground\s+glass':
        name= PATH_TO_SAVE_CSV+'Lung_ground_glass.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    elif dis=='air\s+trapping':
        name= PATH_TO_SAVE_CSV+'Lung_air_trapping.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    elif dis=='pleural\s+effusion':
        name= PATH_TO_SAVE_CSV+'Lung_pleural_effusion.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)

    elif dis=='architectural\s+distortion':
        name= PATH_TO_SAVE_CSV+'Lung_architectural_distortion.csv'
        cases = lung_dict[dis]
        Writting_disease_CSV(lst=cases,main_lst=report_list,Name=name)
    else:
        name=PATH_TO_SAVE_CSV+'Lung_'+dis+'.csv'
        cases = lung_dict[dis]
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
Greater_than_Threshold_disease = [word.replace('surgically\\s+absent','surgically_absent') for word in Greater_than_Threshold_disease]
Greater_than_Threshold_disease = [word.replace('fatty\\s+infiltr','fatty_infiltr') for word in Greater_than_Threshold_disease]
Greater_than_Threshold_disease = [word.replace('ground\\s+glass','ground_glass') for word in Greater_than_Threshold_disease]
Greater_than_Threshold_disease = [word.replace('air\s+trapping','air_trapping') for word in Greater_than_Threshold_disease]
Greater_than_Threshold_disease = [word.replace('pleural\s+effusion','pleural_effusion') for word in Greater_than_Threshold_disease]
Greater_than_Threshold_disease = [word.replace('architectural\s+distortion','architectural_distortion') for word in Greater_than_Threshold_disease]

for i in range (0, len(Greater_than_Threshold_disease)):
    print('-----------------@@@-{}-@@@@---------------'.format(Greater_than_Threshold_disease[i]))
    lm_list2=[]
    for j in range(0,len(Greater_than_Threshold_disease)):
        first_csv=PATH_TO_SAVE_CSV+'Lung_'+Greater_than_Threshold_disease[i]+'.csv'
        Second_csv=PATH_TO_SAVE_CSV+'Lung_'+Greater_than_Threshold_disease[j]+'.csv'
        im=overall(first_csv,Second_csv)
        lm_list2.append(im)
    lm_list.append(lm_list2)



print("-------------Confussion Matrix-----------")
df_cm = pd.DataFrame(lm_list, index = [i for i in Greater_than_Threshold_disease],columns = [i for i in Greater_than_Threshold_disease])
df_cm.to_csv(PATH_TO_SAVE_CSV+"LungDiagnosus0verlap.csv", encoding='utf-8', index=False)
plt.figure(figsize = (40,40))
sn.heatmap(df_cm, annot=True,fmt="d")
plt.savefig(PATH_TO_SAVE_CSV+'LungDiagnosus0verlap.png')

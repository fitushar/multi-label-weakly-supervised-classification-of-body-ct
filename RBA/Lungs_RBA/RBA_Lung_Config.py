####0000000000000-----INPUTS----------0000
REPORT_CSV="/path/to/report/csv/report.csv"



LUNG_ORGAN_LIST = ['lung',
                  ['lung','pulmonary'],
                  ['lung', 'pulmonary', '(lower|upper|middle)\s+lobe', 'centrilobular', 'perifissural','(left|right)\s+base\s', 'bases', 'basilar', 'bronch', 'trachea', 'airspace', 'airway']]



COMMON_DIS_LIST = ['mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren',
                  'hyperenhanc' , 'hypoenhanc', 'aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct',
                  'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural\s+distortion',
                  'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi',
                  'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent',
                   'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent']

LUNG_DIS_LIST=['pneumothorax', 'emphysema', 'pneumoni', 'ground\s+glass', 'aspiration', 'bronchiectasis', 'atelecta', 'embol', 'fibrosis','air\s+trapping','pleural\s+effusion','pneumonectomy']
ABANDON_LIST = ['postsurgical', 'posttreatment', 'postradiation', 'postoperative', 'cholecystectomy', 'resection', 'cholelithiasis','cystectomy']


PATH_TO_SAVE_CSV="/path/to/save/dictonary/and/data/satatistics/Save_Satatistics/"

####-------|Statistics-----inputs-----------------|----###
LIST_FOR_OVERLAP_STATISTICS=['normal','mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren','pleural\s+effusion',
                  'hyperenhanc' , 'hypoenhanc', 'aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct',
                  'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural\s+distortion',
                  'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi',
                  'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent',
                   'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent','pneumothorax', 'emphysema', 'pneumoni', 'ground\s+glass',
                   'aspiration', 'bronchiectasis', 'atelecta', 'embol', 'fibrosis','air\s+trapping']

DISEASE_NAME_AND_NUMBERS=PATH_TO_SAVE_CSV+"LungDisease_NameAndNumberDecending.csv"
DISEASE_COUNT_THRESHOLD= 1

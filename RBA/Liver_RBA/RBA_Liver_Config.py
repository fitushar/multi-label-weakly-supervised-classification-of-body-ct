

REPORT_CSV="/path/to/report/csv/report.csv"
#-Organ Descriptor
LIVER_ORGAN_LIST= ['liver', ['liver', 'hepatic'], ['liver','hepatic','\w+hepatic', 'hepato','gallbladder', 'thegallbladder','gall\s+bladder','biliary', 'bile' ,'(left|right|caudate|quadrate)\s+lobe']]
#-Multi-organ Descriptor
COMMON_DIS_LIST = [ 'mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren',  'hyperenhanc' , 'hypoenhanc','aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct', 'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural distortion', 'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi', 'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent', 'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent']
#-Single Organ Descriptor
LIVER_DIS_LIST = ['steatosis', 'cirrho','cholecystectomy','gallstone','cholelithiasis']


PATH_TO_SAVE_CSV="/path/to/save/dictonary/and/data/satatistics/Save_Satatistics/"
LIST_FOR_OVERLAP_STATISTICS=['normal','mass','opaci', 'calcul', 'stone', 'scar', 'metas', 'malignan', 'cancer', 'tumor', 'neoplasm', 'lithiasis', 'atroph', 'recurren','aneurysm', 'lesion', 'nodule', 'nodular', 'calcifi', 'opacit', 'effusion', 'resect', 'thromb', 'infect', 'infarct', 'inflam', 'fluid', 'consolidat', 'degenerative', 'dissect', 'collaps', 'fissure', 'edema', 'cyst', 'focus', 'angioma', 'spiculated', 'architectural distortion', 'lytic', 'pathologic', 'defect', 'hernia', 'biops', 'encasement', 'fibroid', 'hemorrhage', 'multilocul', 'distension','distention', 'stricture', 'obstructi', 'hypodens', 'hyperdens', 'hypoattenuat', 'hyperattenuat', 'necrosis', 'irregular', 'ectasia', 'destructi', 'dilat', 'granuloma', 'enlarged', 'abscess', 'stent', 'fatty\s+infiltr', 'stenosis', 'delay', 'carcinoma', 'adenoma', 'atrophy', 'hemangioma', 'density', 'surgically\s+absent' ,'steatosis', 'cirrho','cholecystectomy','gallstone','cholelithiasis']

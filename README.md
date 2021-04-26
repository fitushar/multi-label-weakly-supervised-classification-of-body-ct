# Multi-Disease Classification of 13,667 Body CT Scans Using Weakly Supervised Deep Learning

A rule-based algorithm enabled automatic extraction of disease labels from tens of thousands of radiology reports. These weak labels were used to create deep learning models to classify multiple diseases for three different organ systems in body CT. This Repo contains the updated implementation of our paper **"Multi-Disease Classification of 13,667 Body CT ScansUsing Weakly-Supervised Deep Learning"** (Under-review). A pre-print is available: **"Weakly Supervised Multi-Organ Multi-Disease Classification of Body CT Scans"**:https://arxiv.org/abs/2008.01158.

### Citation
```ruby
@misc{tushar2020weakly,
      title={Weakly Supervised Multi-Organ Multi-Disease Classification of Body CT Scans}, 
      author={Fakrul Islam Tushar and Vincent M. D'Anniballe and Rui Hou and Maciej A. Mazurowski and Wanyi Fu and Ehsan Samei 
      and Geoffrey D. Rubin and Joseph Y. Lo},
      year={2020},
      eprint={2008.01158},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```ruby
Tushar, F.I., D'Anniballe, V.M., Hou, R., Mazurowski, M.A., Fu, W., Samei, E., Rubin, G.D., Lo, J.Y., 
2020. Weakly Supervised Multi-Organ Multi-Disease Classification of Body CT Scans. arXiv preprint arXiv:2008.01158.
```

### Abstract

**Background:** Training deep learning classifiers typically requires massive amounts of manual annotation. Weak supervision may
leverage existing medical data to classify multiple organ systems and diseases.

**Purpose:** To design multi-disease classifiers for body computed tomography (CT) scans using automatically extracted labels
from radiology text reports.


**Materials & Methods:** This retrospective study deployed rule-based algorithms to extract 19,255 disease labels from reports of 13,667
body CT scans of 12,092 subjects for training. Using a 3D DenseVNet, three organ systems were segmented:
lungs/pleura, liver/gallbladder, and kidneys/ureters. For each organ, a 3D convolutional neural network classified
normality versus four common diseases. Testing was performed on an additional 2,158 CT volumes relative to 2,875
manually derived reference labels.

**Results:** Manual validation of the extracted labels confirmed 91 to 99% accuracy. Performance using the receiver
operating characteristic area under the curve (AUC) for lungs/pleura labels were as follows: atelectasis 0.77 (95%
CI: 0.74 to 0.81), nodule 0.65 (0.61 to 0.69), emphysema 0.89 (0.86 to 0.92), effusion 0.97 (0.96 to 0.98), and
normal 0.89 (0.87 to 0.91). For liver/gallbladder: stone 0.62 (0.56 to 0.67), lesion 0.73 (0.69 to 0.77), dilation 0.87
(0.84 to 0.90), fatty 0.89 (0.86 to 0.92), and normal 0.82 (0.78 to 0.85). For kidneys/ureters: stone 0.83 (0.79 to
0.87), atrophy 0.92 (0.89 to 0.94), lesion 0.68 (0.64 to 0.72), cyst 0.70 (0.66 to 0.73), and normal 0.79 (0.75 to 0.83).

**Conclusion:** Weakly supervised deep learning classifiers leveraged massive amounts of unannotated body CT data to classify
multiple organ systems and diseases.

### Results
<img src="ReadMeFigure/result.png"  width="40%" height="40%"><img src="ReadMeFigure/Probabilities.png"  width="60%" height="60%">


# Study Overview:
This Study contains three main modules:

* **1.**  Developed and Used RBA to generate labels from radiology reports.
* **2.** Segmentation Module to extracted segmentation mask from the CT volume and
* **3.** Classification Module With three independent classifiers to classify disease in lungs/pleura, liver/gallbladder, and kidneys/ureters.


# Rule-Based Algorithm (RBA):
In this section, we describe and outline the development processes of our RBA.

A radiology CT report contains a protocol, indication, technique, findings, and impression sections. RBA was limited to the findings section of the CT reports minimizing the influence of biasing information referenced in other sections and ensuring that the automated annotation reflected image information in the current exam (e.g., indication for the exam, patient history, technique factors, and comparison with priors). For example, the impression section could describe a diagnosis based on patient history that could not be made using solely image-based information.

| ![report_text.PNG](ReadMeFigure/report_text.PNG) | 
|:--:| 
| *Representative example of a body CT radiology report within our dataset.* |


We used a dictionary approach to develop RBAs to extract disease labels from radiology text reports. To select target disease and organ keywords for the RBA dictionary, we computed term frequency-inverse document frequency (TF-IDF) on the findings sections of a random batch of 3,500 radiology reports. A board-certified radiologist guided to define the TF-IDF terms into several categories, specifically: 
* a) single-organ descriptors specific to each organ, e.g., pleural effusion or steatosis, 
* b) multi-organ descriptors applicable to numerous organs, e.g., nodule or stone, 
* c) negation terms indicating the absence of disease, e.g., no or without, 
* d) qualifier terms describing confounding conditions, e.g., however, OR 
* e) normal terms suggesting normal anatomy in the absence of other diseases and abnormalities, e.g., unremarkable. 

The figure below displays the dictionary terms and their descriptor type for each organ system.
<img src="ReadMeFigure/Dictionary.PNG"  width="80%" height="80%"> 

Figure 4 displays an overview of the RBA’s flowchart and logic. Although a separate RBA was created for each organ system, the workflow was the same. After the dictionary was refined, report text was converted to lowercase, and each sentence was tokenized. In summary, the RBA was deployed on each sentence, and the number of potential diseases was counted first using the logic for the multi-organ descriptor and then the single-organ descriptor. If no potential disease labels were detected, then the normal descriptor logic was finally applied to verify normality. This process was repeated for each disease outcome allowing a report to be positive for one or more diseases or normal. Note that in this study an organ system was defined as normal not only by excluding the four diseases studied but also in the absence of dozens of abnormalities and diseases states that were not otherwise analyzed, as shown in Appendix 1. If the RBA failed to categorize the report definitively as positive for disease or normal (e.g., there was no mention of the organ system), then the report was labeled as uncertain and was not included in this study.


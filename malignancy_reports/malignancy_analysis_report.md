# Breast MRI Malignancy Analysis Report

## Dataset Statistics

- **Total Cases**: 651 patients
- **Total UIDs**: 1302 bilateral assessments
- **Malignant Cases**: 679
- **Benign Cases**: 623

## Malignancy Distribution by Side

| Side   |   BENIGN |   MALIGNANT |
|:-------|---------:|------------:|
| left   |      322 |         329 |
| right  |      301 |         350 |

## Summary Table (Sample)

|   Patient_ID | UID       | Side   |   Malignant | Malignancy_Status   |   Fold | Split   | Clinical_Note                                                                                                                                                                            |
|-------------:|:----------|:-------|------------:|:--------------------|-------:|:--------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          001 | 001_left  | left   |           1 | MALIGNANT           |      4 | train   | Findings in the left breast are highly suspicious for malignancy, showing irregular margins and rapid contrast uptake. Urgent biopsy is recommended for histopathological analysis.      |
|          001 | 001_right | right  |           0 | BENIGN              |      4 | train   | Assessment of the right breast is unremarkable, with benign-appearing parenchyma. Continued routine screening is appropriate.                                                            |
|          002 | 002_left  | left   |           1 | MALIGNANT           |      4 | test    | Suspicious enhancement patterns detected in left breast. Irregular morphological characteristics suggest malignancy. Requires immediate biopsy confirmation and oncological evaluation.  |
|          002 | 002_right | right  |           0 | BENIGN              |      4 | test    | Imaging of the right breast reveals normal physiological enhancement with no suspicious lesions identified. The findings are categorized as benign.                                      |
|          004 | 004_left  | left   |           1 | MALIGNANT           |      4 | train   | A mass with suspicious morphology and concerning enhancement kinetics has been identified in the left breast. Follow-up with surgical oncology is advised for definitive diagnosis.      |
|          004 | 004_right | right  |           0 | BENIGN              |      4 | train   | Assessment of the right breast is unremarkable, with benign-appearing parenchyma. Continued routine screening is appropriate.                                                            |
|          005 | 005_left  | left   |           0 | BENIGN              |      4 | train   | The left breast shows no evidence of malignancy. Observed enhancement is consistent with benign fibrocystic changes. Recommend routine follow-up imaging as per standard guidelines.     |
|          005 | 005_right | right  |           1 | MALIGNANT           |      4 | train   | Suspicious enhancement patterns detected in right breast. Irregular morphological characteristics suggest malignancy. Requires immediate biopsy confirmation and oncological evaluation. |
|          007 | 007_left  | left   |           1 | MALIGNANT           |      4 | test    | Suspicious enhancement patterns detected in left breast. Irregular morphological characteristics suggest malignancy. Requires immediate biopsy confirmation and oncological evaluation.  |
|          007 | 007_right | right  |           0 | BENIGN              |      4 | test    | Assessment of the right breast is unremarkable, with benign-appearing parenchyma. Continued routine screening is appropriate.                                                            |
|          008 | 008_left  | left   |           0 | BENIGN              |      4 | train   | The left breast shows no evidence of malignancy. Observed enhancement is consistent with benign fibrocystic changes. Recommend routine follow-up imaging as per standard guidelines.     |
|          008 | 008_right | right  |           1 | MALIGNANT           |      4 | train   | Findings in the right breast are highly suspicious for malignancy, showing irregular margins and rapid contrast uptake. Urgent biopsy is recommended for histopathological analysis.     |
|          009 | 009_left  | left   |           0 | BENIGN              |      4 | val     | Normal parenchymal enhancement in left breast. Benign morphological patterns observed. Routine surveillance recommended.                                                                 |
|          009 | 009_right | right  |           1 | MALIGNANT           |      4 | val     | Evaluation of the right breast reveals abnormal enhancement patterns consistent with a malignant tumor. Biopsy is required for definitive diagnosis and treatment planning.              |
|          010 | 010_left  | left   |           1 | MALIGNANT           |      4 | train   | Suspicious enhancement patterns detected in left breast. Irregular morphological characteristics suggest malignancy. Requires immediate biopsy confirmation and oncological evaluation.  |
|          010 | 010_right | right  |           0 | BENIGN              |      4 | train   | The findings in the right breast are benign in nature, with no features to suggest malignancy. Standard clinical and imaging surveillance is recommended.                                |
|          012 | 012_left  | left   |           0 | BENIGN              |      4 | train   | Imaging of the left breast reveals normal physiological enhancement with no suspicious lesions identified. The findings are categorized as benign.                                       |
|          012 | 012_right | right  |           1 | MALIGNANT           |      4 | train   | Suspicious enhancement patterns detected in right breast. Irregular morphological characteristics suggest malignancy. Requires immediate biopsy confirmation and oncological evaluation. |
|          013 | 013_left  | left   |           1 | MALIGNANT           |      4 | test    | A mass with suspicious morphology and concerning enhancement kinetics has been identified in the left breast. Follow-up with surgical oncology is advised for definitive diagnosis.      |
|          013 | 013_right | right  |           0 | BENIGN              |      4 | test    | Normal parenchymal enhancement in right breast. Benign morphological patterns observed. Routine surveillance recommended.                                                                |


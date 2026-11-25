# NHANES Data Download Instructions

Raw National Health and Nutrition Examination Survey (NHANES) files are publicly distributed by the U.S. Centers for Disease Control and Prevention (CDC) / National Center for Health Statistics (NCHS). In accordance with NCHS distribution guidelines, raw `.XPT` files are **not redistributed** in this repository.

Download the official SAS transport (`.XPT`) files for both survey cycles directly from the CDC NHANES portal:

- **2017-2018 Cycle (`_J`):** https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
- **August 2021-August 2023 Cycle (`_L`):** https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

Place all 32 downloaded files inside `data/nhanes/`:

| Component | 2017-2018 (`_J`) | Aug. 2021-Aug. 2023 (`_L`) |
| :--- | :--- | :--- |
| Demographics | `DEMO_J.xpt` | `DEMO_L.xpt` |
| Body Measures | `BMX_J.xpt` | `BMX_L.xpt` |
| Oscillometric Blood Pressure | `BPXO_J.xpt` | `BPXO_L.xpt` |
| Liver Ultrasound Transient Elastography | `LUX_J.xpt` | `LUX_L.xpt` |
| Standard Biochemistry Profile | `BIOPRO_J.xpt` | `BIOPRO_L.xpt` |
| Glycohemoglobin (HbA1c) | `GHB_J.xpt` | `GHB_L.xpt` |
| Plasma Fasting Glucose | `GLU_J.xpt` | `GLU_L.xpt` |
| Albumin & Creatinine - Urine | `ALB_CR_J.xpt` | `ALB_CR_L.xpt` |
| Cholesterol - LDL & Triglycerides | `TRIGLY_J.xpt` | `TRIGLY_L.xpt` |
| Cholesterol - Total | `TCHOL_J.xpt` | `TCHOL_L.xpt` |
| Cholesterol - HDL | `HDL_J.xpt` | `HDL_L.xpt` |
| Complete Blood Count | `CBC_J.xpt` | `CBC_L.xpt` |
| Diabetes Questionnaire | `DIQ_J.xpt` | `DIQ_L.xpt` |
| Alcohol Use Questionnaire | `ALQ_J.xpt` | `ALQ_L.xpt` |
| Hepatitis B Surface Antigen | `HEPBD_J.xpt` | `HEPBD_L.xpt` |
| Hepatitis C Confirmed Antibody & RNA | `HEPC_J.xpt` | `HEPC_L.xpt` |

> **Cycle Harmonization Note:** In the August 2021-August 2023 `TRIGLY_L.xpt` release, serum triglycerides (mg/dL) are stored under `LBXTLG`. The loader in `src/cohorts.py` automatically maps `LBXTLG -> LBXTR` so both survey periods share identical predictor sets.

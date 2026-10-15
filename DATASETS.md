# Dataset features and targets

## SUPPORT

Dataset label: `support_withoutmissing_outputs`. 48 inputs, 4 targets.

Input location: `datasets/support_withoutmissing_outputs/support_withoutmissing_outputs.csv`.

One row per patient with all four targets present. Features include demographics, diagnoses, day-3 physiological/laboratory measures and average treatment intensity over days 3–25, they are not all admission-time observations. Categorical suffixes are literal source levels, _nan denotes a missing-category indicator.

### Inputs

| Column | Description |
| --- | --- |
| `age` | Patient age in years. |
| `sex` | Recorded sex: 0 = male, 1 = female. |
| `num.co` | Number of comorbidities. |
| `edu` | Years of education. |
| `scoma` | SUPPORT coma score based on the day-3 Glasgow assessment. |
| `avtisst` | Average Therapeutic Intervention Scoring System score over days 3–25. |
| `meanbp` | Mean arterial blood pressure on day 3. |
| `wblc` | White blood cell count on day 3. |
| `hrt` | Heart rate on day 3. |
| `resp` | Respiratory rate on day 3. |
| `temp` | Body temperature on day 3 (°C). |
| `pafi` | Day-3 PaO2/FiO2 oxygenation ratio, using the source's FiO2 convention. |
| `alb` | Serum albumin on day 3. |
| `bili` | Bilirubin on day 3. |
| `crea` | Serum creatinine on day 3. |
| `sod` | Serum sodium on day 3. |
| `ph` | Arterial pH on day 3. |
| `glucose` | Blood glucose on day 3. |
| `bun` | Blood urea nitrogen on day 3. |
| `urine` | Urine output on day 3. |
| `adlp` | Day-3 activities-of-daily-living score reported by the patient. |
| `adls` | Day-3 activities-of-daily-living score reported by a surrogate. |
| `adlsc` | Source-imputed activities-of-daily-living score calibrated to the surrogate scale. |
| `dzgroup_ARF/MOSF w/Sepsis` | Diagnosis-group indicator: acute respiratory failure or multiple-organ-system failure with sepsis. |
| `dzgroup_CHF` | Diagnosis-group indicator: congestive heart failure. |
| `dzgroup_COPD` | Diagnosis-group indicator: chronic obstructive pulmonary disease. |
| `dzgroup_Cirrhosis` | Diagnosis-group indicator: cirrhosis. |
| `dzgroup_Colon Cancer` | Diagnosis-group indicator: colon cancer. |
| `dzgroup_Coma` | Diagnosis-group indicator: coma. |
| `dzgroup_Lung Cancer` | Diagnosis-group indicator: lung cancer. |
| `dzgroup_MOSF w/Malig` | Diagnosis-group indicator: multiple-organ-system failure with malignancy. |
| `dzgroup_nan` | Indicator that the source dzgroup category is missing. |
| `dzclass_ARF/MOSF` | Diagnosis-class indicator: acute respiratory failure or multiple-organ-system failure. |
| `dzclass_COPD/CHF/Cirrhosis` | Diagnosis-class indicator: chronic obstructive pulmonary disease, congestive heart failure or cirrhosis. |
| `dzclass_Cancer` | Diagnosis-class indicator: cancer. |
| `dzclass_Coma` | Diagnosis-class indicator: coma. |
| `dzclass_nan` | Indicator that the source dzclass category is missing. |
| `income_$11-$25k` | Income-bracket indicator: $11-$25k. |
| `income_$25-$50k` | Income-bracket indicator: $25-$50k. |
| `income_>$50k` | Income-bracket indicator: >$50k. |
| `income_under $11k` | Income-bracket indicator: under $11k. |
| `income_nan` | Indicator that the source income category is missing. |
| `race_asian` | Recorded race/ethnicity indicator: asian. |
| `race_black` | Recorded race/ethnicity indicator: black. |
| `race_hispanic` | Recorded race/ethnicity indicator: hispanic. |
| `race_other` | Recorded race/ethnicity indicator: other. |
| `race_white` | Recorded race/ethnicity indicator: white. |
| `race_nan` | Indicator that the source race category is missing. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_hospdead` | Binary | In-hospital death: 1 = died, 0 = survived hospitalization. |
| `target_regression_sfdm2` | Regression | Source functional-status category encoded as 1–5, code 5 means less than two months of follow-up. |
| `target_regression_charges` | Regression | Hospital charges in source monetary units. |
| `target_binary_death` | Binary | Death observed during source follow-up, up to the National Death Index cutoff of 31 December 1994. |

Functional-status coding: 1 = `no(M2 and SIP pres)`, 2 = `adl>=4 (>=5 if sur)`, 3 = `SIP>=30`, 4 = `Coma or Intub`, 5 = `<2 mo. follow-up`. This is the implemented source-category encoding, not a clean severity continuum.

## SUPPORT2

Dataset label: `support2_withoutmissing_outputs`. 56 inputs, 4 targets.

Input location: `datasets/support2_withoutmissing_outputs/support2_withoutmissing_outputs.csv`.

Larger SUPPORT release, with all four targets present. Adds physiology scores, diabetes, dementia and cancer indicators to the shared demographic/clinical schema. Inputs have mixed observation times, including treatment intensity over days 3–25, _nan denotes a missing-category indicator.

### Inputs

| Column | Description |
| --- | --- |
| `age` | Patient age in years. |
| `sex` | Recorded sex: 0 = male, 1 = female. |
| `num.co` | Number of comorbidities. |
| `edu` | Years of education. |
| `scoma` | SUPPORT coma score based on the day-3 Glasgow assessment. |
| `avtisst` | Average Therapeutic Intervention Scoring System score over days 3–25. |
| `sps` | SUPPORT physiology score on day 3. |
| `aps` | Day-3 APACHE III acute physiology score, excluding coma, as supplied by the source. |
| `diabetes` | Indicator of diabetes. |
| `dementia` | Indicator of dementia. |
| `meanbp` | Mean arterial blood pressure on day 3. |
| `wblc` | White blood cell count on day 3. |
| `hrt` | Heart rate on day 3. |
| `resp` | Respiratory rate on day 3. |
| `temp` | Body temperature on day 3 (°C). |
| `pafi` | Day-3 PaO2/FiO2 oxygenation ratio, using the source's FiO2 convention. |
| `alb` | Serum albumin on day 3. |
| `bili` | Bilirubin on day 3. |
| `crea` | Serum creatinine on day 3. |
| `sod` | Serum sodium on day 3. |
| `ph` | Arterial pH on day 3. |
| `glucose` | Blood glucose on day 3. |
| `bun` | Blood urea nitrogen on day 3. |
| `urine` | Urine output on day 3. |
| `adlp` | Day-3 activities-of-daily-living score reported by the patient. |
| `adls` | Day-3 activities-of-daily-living score reported by a surrogate. |
| `adlsc` | Source-imputed activities-of-daily-living score calibrated to the surrogate scale. |
| `dzgroup_ARF/MOSF w/Sepsis` | Diagnosis-group indicator: acute respiratory failure or multiple-organ-system failure with sepsis. |
| `dzgroup_CHF` | Diagnosis-group indicator: congestive heart failure. |
| `dzgroup_COPD` | Diagnosis-group indicator: chronic obstructive pulmonary disease. |
| `dzgroup_Cirrhosis` | Diagnosis-group indicator: cirrhosis. |
| `dzgroup_Colon Cancer` | Diagnosis-group indicator: colon cancer. |
| `dzgroup_Coma` | Diagnosis-group indicator: coma. |
| `dzgroup_Lung Cancer` | Diagnosis-group indicator: lung cancer. |
| `dzgroup_MOSF w/Malig` | Diagnosis-group indicator: multiple-organ-system failure with malignancy. |
| `dzgroup_nan` | Indicator that the source dzgroup category is missing. |
| `dzclass_ARF/MOSF` | Diagnosis-class indicator: acute respiratory failure or multiple-organ-system failure. |
| `dzclass_COPD/CHF/Cirrhosis` | Diagnosis-class indicator: chronic obstructive pulmonary disease, congestive heart failure or cirrhosis. |
| `dzclass_Cancer` | Diagnosis-class indicator: cancer. |
| `dzclass_Coma` | Diagnosis-class indicator: coma. |
| `dzclass_nan` | Indicator that the source dzclass category is missing. |
| `income_$11-$25k` | Income-bracket indicator: $11-$25k. |
| `income_$25-$50k` | Income-bracket indicator: $25-$50k. |
| `income_>$50k` | Income-bracket indicator: >$50k. |
| `income_under $11k` | Income-bracket indicator: under $11k. |
| `income_nan` | Indicator that the source income category is missing. |
| `race_asian` | Recorded race/ethnicity indicator: asian. |
| `race_black` | Recorded race/ethnicity indicator: black. |
| `race_hispanic` | Recorded race/ethnicity indicator: hispanic. |
| `race_other` | Recorded race/ethnicity indicator: other. |
| `race_white` | Recorded race/ethnicity indicator: white. |
| `race_nan` | Indicator that the source race category is missing. |
| `ca_metastatic` | Cancer-status indicator: metastatic. |
| `ca_no` | Cancer-status indicator: no. |
| `ca_yes` | Cancer-status indicator: yes. |
| `ca_nan` | Indicator that the source ca category is missing. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_hospdead` | Binary | In-hospital death: 1 = died, 0 = survived hospitalization. |
| `target_regression_sfdm2` | Regression | Source functional-status category encoded as 1–5, code 5 means less than two months of follow-up. |
| `target_regression_charges` | Regression | Hospital charges in source monetary units. |
| `target_binary_death` | Binary | Death observed during source follow-up, up to the National Death Index cutoff of 31 December 1994. |

Functional-status coding: 1 = `no(M2 and SIP pres)`, 2 = `adl>=4 (>=5 if sur)`, 3 = `SIP>=30`, 4 = `Coma or Intub`, 5 = `<2 mo. follow-up`. This is the implemented source-category encoding, not a clean severity continuum.

## WHAS1

Dataset label: `whas1_preprocessed`. 9 inputs, 2 targets.

Input location: `datasets/whas1_preprocessed/whas1_preprocessed.csv`.

One row per myocardial-infarction admission in the 481-record first-edition Worcester Heart Attack Study release. Infarction type is one-hot encoded.

### Inputs

| Column | Description |
| --- | --- |
| `AGE` | Age in years. |
| `SEX` | Recorded sex: 0 = male, 1 = female. |
| `CPK` | Peak cardiac enzyme level, in international units. |
| `SHO` | Cardiogenic shock complications: 1 = present. |
| `CHF` | Left-heart-failure complications: 1 = present. |
| `MIORD` | Myocardial infarction order: 0 = first, 1 = recurrent. |
| `MITYPE_1` | Indicator of Q-wave myocardial infarction. |
| `MITYPE_2` | Indicator of non-Q-wave myocardial infarction. |
| `MITYPE_3` | Indicator of indeterminate myocardial infarction type. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_regression_LENSTAY` | Regression | Length of hospital stay in days. |
| `target_binary_DSTAT` | Binary | Hospital discharge status: 0 = alive, 1 = dead. |

## WHAS500

Dataset label: `whas500_preprocessed`. 13 inputs, 2 targets.

Input location: `datasets/whas500_preprocessed/whas500_preprocessed.csv`.

One row per myocardial-infarction admission in the 500-record Worcester Heart Attack Study release. Infarction type remains a native binary input.

### Inputs

| Column | Description |
| --- | --- |
| `age` | Age at hospital admission, in years. |
| `gender` | Recorded gender: 0 = male, 1 = female. |
| `hr` | Initial heart rate, in beats per minute. |
| `sysbp` | Initial systolic blood pressure, in mmHg. |
| `diasbp` | Initial diastolic blood pressure, in mmHg. |
| `bmi` | Body mass index, in kg/m². |
| `cvd` | History of cardiovascular disease: 1 = present. |
| `afb` | Atrial fibrillation: 1 = present. |
| `sho` | Cardiogenic shock: 1 = present. |
| `chf` | Congestive heart complications: 1 = present. |
| `av3` | Complete heart block: 1 = present. |
| `miord` | Myocardial infarction order: 0 = first, 1 = recurrent. |
| `mitype` | Myocardial infarction type: 0 = non-Q-wave, 1 = Q-wave. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_regression_los` | Regression | Length of hospital stay in days, from admission to discharge. |
| `target_binary_dstat` | Binary | Hospital discharge status: 0 = alive, 1 = dead. |
## ASSISTments 2009–2010 — SB

Dataset label: `assistments2009_sb`. 78 inputs, 3 targets.

Input location: `datasets/assistments2009-2010/prepared/assistments2009_sb`.

One row per student, predicting the reserved final Skill Builder assignment from earlier ended assignments.

### Inputs

| Column | Description |
| --- | --- |
| `original_problem_count` | Number of historical rows with original = 1. |
| `ended_assignment_count` | Number of earlier SB assignments that ended before the target assignment started. |
| `scaffolding_problem_rate` | Fraction of historical interaction rows with original = 0 (scaffolding). |
| `interactions` | Number of all historical interaction rows, including scaffolding. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `correct_rate` | Correct-response fraction among historical original-problem rows. |
| `attempt_count_mean` | Mean of attempt counts across historical original-problem rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical original-problem rows. |
| `attempt_count_total` | Sum of attempt counts across historical original-problem rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical original-problem rows. |
| `hint_count_mean` | Mean hint count among historical original-problem rows. |
| `hint_count_total` | Total hint count across historical original-problem rows. |
| `hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0. |
| `bottom_hint_rate` | Fraction of historical original-problem rows with bottom_hint = 1 (final hint). |
| `first_action_attempt_rate` | Fraction of historical original-problem rows with an attempt as the first action (first_action = 0). |
| `first_action_hint_rate` | Fraction of historical original-problem rows with a hint as the first action (first_action = 1). |
| `first_action_scaffolding_rate` | Fraction of historical original-problem rows with scaffolding as the first action (first_action = 2). |
| `overlap_time_mean_seconds` | Mean of recorded interaction time across all historical interaction rows, including scaffolding. |
| `overlap_time_std_seconds` | Standard deviation of recorded interaction time across all historical interaction rows, including scaffolding. |
| `overlap_time_total_seconds` | Sum of recorded interaction time across all historical interaction rows, including scaffolding. |
| `overlap_time_p90_seconds` | 90th percentile of recorded interaction time across all historical interaction rows, including scaffolding. |
| `answer_type_algebra_rate` | Fraction of historical original-problem rows with algebra response format. |
| `answer_type_algebra_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_total_seconds` | Sum of recorded interaction time across historical original-problem rows, restricted to algebra response format. |
| `answer_type_choose_1_rate` | Fraction of historical original-problem rows with single-choice response format. |
| `answer_type_choose_1_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_total_seconds` | Sum of recorded interaction time across historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_n_rate` | Fraction of historical original-problem rows with multiple-selection response format. |
| `answer_type_choose_n_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_total_seconds` | Sum of recorded interaction time across historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_fill_in_1_rate` | Fraction of historical original-problem rows with single-field fill-in response format. |
| `answer_type_fill_in_1_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_total_seconds` | Sum of recorded interaction time across historical original-problem rows, restricted to single-field fill-in response format. |
| `mastery_rate` | Fraction of historical assignments containing any run of three consecutive correct original problems. |
| `assignment_problem_count_mean` | Mean of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_problem_count_std` | Standard deviation of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_problem_count_p90` | 90th percentile of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_mean` | Mean of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_std` | Standard deviation of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_p90` | 90th percentile of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `previous_assignment_mastered` | Any-three mastery indicator of the most recently ended historical assignment. |
| `previous_assignment_problem_count` | Distinct problem_id count, including scaffolding, in the most recently ended historical assignment. |
| `previous_assignment_time` | Sum of recorded interaction work time in the most recently ended historical assignment, including scaffolding. |
| `selected_assignment_answer_type_algebra_present` | Indicator that any retained target-assignment row has algebra response format. |
| `selected_assignment_answer_type_choose_1_present` | Indicator that any retained target-assignment row has single-choice response format. |
| `selected_assignment_answer_type_choose_n_present` | Indicator that any retained target-assignment row has multiple-selection response format. |
| `selected_assignment_answer_type_fill_in_1_present` | Indicator that any retained target-assignment row has single-field fill-in response format. |
| `attempt_count_median` | Median of attempt counts across historical original-problem rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical original-problem rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical original-problem rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical original-problem rows. |
| `overlap_time_median_seconds` | Median of recorded interaction time across historical original-problem rows. |
| `overlap_time_p75_seconds` | 75th percentile of recorded interaction time across historical original-problem rows. |
| `overlap_time_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction time across historical original-problem rows. |
| `overlap_time_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction time across historical original-problem rows. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical original-problem rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_overlap_time_median_seconds` | Median of recorded interaction time across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_overlap_time_p90_seconds` | 90th percentile of recorded interaction time across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `selected_skill_prior_interactions` | Number of earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_has_history` | Indicator that there is at least one of the earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_attempt_count_mean` | Mean of attempt counts across historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_overlap_time_mean_seconds` | Mean of recorded interaction time across historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical original-problem rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical original-problem rows. |
| `recent_10_log1p_overlap_time_seconds_trend` | OLS slope of log1p(interaction time in seconds) against chronological row position in up to ten latest historical original-problem rows. |
| `recent_10_hint_use_trend` | OLS slope of hint-use indicator (hint_count > 0) against chronological row position in up to ten latest historical original-problem rows. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_mastered` | Binary | Any run of three consecutive correct original problems within the target assignment. |
| `target_regression_problems` | Regression | Distinct problem_id count across all retained target-assignment rows, including scaffolding. |
| `target_regression_work_time` | Regression | Sum of recorded work time over all retained target-assignment interactions, including scaffolding. |

## ASSISTments 2009–2010 — NSB

Dataset label: `assistments2009_nsb`. 85 inputs, 3 targets.

Input location: `datasets/assistments2009-2010/prepared/assistments2009_nsb`.

One row per student, predicting the final retained non-Skill-Builder problem interaction from earlier interactions. last_interaction_* describes the selected target problem, not the previous historical problem.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `scaffolding_problem_rate` | Fraction of historical interaction rows with original = 0. |
| `original_problem_count` | Number of historical rows with original = 1. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `hint_count_mean` | Mean hint count among historical interaction rows. |
| `hint_count_total` | Total hint count across historical interaction rows. |
| `hint_use_rate` | Fraction of historical interaction rows with hint_count > 0. |
| `bottom_hint_rate` | Fraction of historical interaction rows with bottom_hint = 1 (final hint). |
| `first_action_attempt_rate` | Fraction of historical interaction rows with an attempt as the first action (first_action = 0). |
| `first_action_hint_rate` | Fraction of historical interaction rows with a hint as the first action (first_action = 1). |
| `first_action_scaffolding_rate` | Fraction of historical interaction rows with scaffolding as the first action (first_action = 2). |
| `last_interaction_original` | Original-problem indicator of the reserved target interaction. |
| `last_interaction_tutor_mode_tutor` | Indicator of tutor mode tutor in the reserved target interaction. |
| `last_interaction_tutor_mode_test` | Indicator of tutor mode test in the reserved target interaction. |
| `last_interaction_tutor_mode_pre` | Indicator of tutor mode pre in the reserved target interaction. |
| `last_interaction_tutor_mode_post` | Indicator of tutor mode post in the reserved target interaction. |
| `last_interaction_answer_type_algebra` | Indicator of response format algebra in the reserved target interaction. |
| `last_interaction_answer_type_choose_1` | Indicator of response format single-choice in the reserved target interaction. |
| `last_interaction_answer_type_choose_n` | Indicator of response format multiple-selection in the reserved target interaction. |
| `last_interaction_answer_type_fill_in_1` | Indicator of response format single-field fill-in in the reserved target interaction. |
| `last_interaction_answer_type_other` | Indicator of response format other or missing in the reserved target interaction. |
| `last_interaction_problem_set_type_linear` | Indicator of problem-set type linear in the reserved target interaction. |
| `last_interaction_problem_set_type_random_iterate` | Indicator of problem-set type random_iterate in the reserved target interaction. |
| `last_interaction_problem_set_type_random_child_order` | Indicator of problem-set type random_child_order in the reserved target interaction. |
| `last_interaction_problem_set_type_numeric_limit` | Indicator of problem-set type numeric_limit in the reserved target interaction. |
| `last_interaction_problem_set_type_choose_condition` | Indicator of problem-set type choose_condition in the reserved target interaction. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows. |
| `overlap_time_std_seconds` | Standard deviation of recorded interaction time across historical interaction rows. |
| `overlap_time_total_seconds` | Sum of recorded interaction time across historical interaction rows. |
| `overlap_time_p90_seconds` | 90th percentile of recorded interaction time across historical interaction rows. |
| `answer_type_algebra_rate` | Fraction of historical interaction rows with algebra response format. |
| `answer_type_choose_1_rate` | Fraction of historical interaction rows with single-choice response format. |
| `answer_type_choose_n_rate` | Fraction of historical interaction rows with multiple-selection response format. |
| `answer_type_fill_in_1_rate` | Fraction of historical interaction rows with single-field fill-in response format. |
| `answer_type_algebra_correct_rate` | Correct-response fraction among historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_total_seconds` | Sum of recorded interaction time across historical interaction rows, restricted to algebra response format. |
| `answer_type_choose_1_correct_rate` | Correct-response fraction among historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_total_seconds` | Sum of recorded interaction time across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_n_correct_rate` | Correct-response fraction among historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_total_seconds` | Sum of recorded interaction time across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_fill_in_1_correct_rate` | Correct-response fraction among historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_total_seconds` | Sum of recorded interaction time across historical interaction rows, restricted to single-field fill-in response format. |
| `previous_problem_correct` | Correctness of the immediately preceding historical interaction. |
| `previous_problem_attempt_count` | Attempt count of the immediately preceding historical interaction. |
| `previous_problem_overlap_time_seconds` | Interaction time of the immediately preceding historical interaction, in seconds. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_hint_use_rate` | Fraction of historical interaction rows with hint_count > 0, using only the up to five latest historical interaction rows. |
| `selected_skill_prior_interactions` | Number of earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_correct_rate` | Correct-response fraction among historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_overlap_time_mean_seconds` | Mean of recorded interaction time across historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_hint_use_rate` | Fraction of historical interaction rows with hint_count > 0, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `overlap_time_median_seconds` | Median of recorded interaction time across historical interaction rows. |
| `overlap_time_p75_seconds` | 75th percentile of recorded interaction time across historical interaction rows. |
| `overlap_time_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction time across historical interaction rows. |
| `overlap_time_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction time across historical interaction rows. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_median_seconds` | Median of recorded interaction time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_p90_seconds` | 90th percentile of recorded interaction time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `selected_skill_has_history` | Indicator that there is at least one of the earlier interactions whose skill intersects the target problem’s skills. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_overlap_time_seconds_trend` | OLS slope of log1p(interaction time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_hint_use_trend` | OLS slope of hint-use indicator (hint_count > 0) against chronological row position in up to ten latest historical interaction rows. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_correct` | Binary | Recorded correctness of the reserved final target interaction. |
| `target_regression_overlap_time` | Regression | Target-interaction overlap_time converted from milliseconds to seconds. |
| `target_regression_attempts` | Regression | Recorded attempt_count of the target interaction. |

## ASSISTments Goldstein P(J), 2011

Dataset label: `assistments_paper_2011`. 70 inputs, 2 targets.

Input location: `datasets/assistments_paper:2011_goldstein/prepared/assistments_paper_2011`.

One row per student, predicting first correctness and first-response time of the reserved final clean problem interaction. History contains interactions ended before target start. Action-group maxima reconstruct attempts and hint counts. Selected-target context is distinct from historical outcomes, P(J) model scores are not inputs or targets here.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical problem interactions ending before the selected target starts. |
| `repeated_problem_rate` | Historical repetition fraction: 1 − distinct problemId count / interaction count. |
| `scaffolding_problem_rate` | Fraction of historical interactions with original = 0. |
| `original_problem_count` | Number of historical interaction rows with original = 1. |
| `correct_rate` | Historical first-response correctness fraction: sum(correct)/interaction count. |
| `hint_count_mean` | Mean hints per historical interaction, each hint count is max(hintCount) over its action group. |
| `hint_count_total` | Sum of historical interaction hint counts reconstructed as max(hintCount). |
| `hint_use_rate` | Fraction of historical interactions with reconstructed hint_count > 0. |
| `bottom_hint_rate` | Fraction of historical interactions with max(bottomHint) = 1, indicating bottom-out hint use. |
| `first_action_attempt_rate` | Fraction of historical interactions starting with neither hint nor scaffolding: hint = 0 and scaffold = 0. |
| `first_action_hint_rate` | Fraction of historical interactions whose first action has hint = 1. |
| `first_action_scaffolding_rate` | Fraction of historical interactions whose first action has scaffold = 1. |
| `last_interaction_original` | Original-problem indicator from the selected target interaction. |
| `during_school` | frWorkingInSchool indicator from the selected target interaction. |
| `problem_type_algebra` | Selected-target problemType = algebra indicator. |
| `problem_type_choose_one` | Selected-target problemType = choose_1 indicator. |
| `problem_type_fill_in` | Selected-target problemType = fill_in_1 indicator. |
| `attempt_count_mean` | Mean of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_std` | Population standard deviation (ddof=0) of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_total` | Sum of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_p90` | 90th percentile of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `first_response_time_mean_seconds` | Mean of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_std_seconds` | Population standard deviation (ddof=0) of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_total_seconds` | Sum of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_p90_seconds` | 90th percentile of first-response time (timeTaken) per historical interaction, in seconds. |
| `answer_type_algebra_rate` | Fraction of all historical interactions with algebra problemType. |
| `answer_type_algebra_correct_rate` | Within historical algebra interactions: first-response correctness fraction. |
| `answer_type_algebra_attempt_count_mean` | Within historical algebra interactions: mean reconstructed attempts. |
| `answer_type_algebra_first_response_time_mean_seconds` | Within historical algebra interactions: mean first-response time in seconds. |
| `answer_type_algebra_first_response_time_total_seconds` | Within historical algebra interactions: total first-response time in seconds. |
| `answer_type_choose_1_rate` | Fraction of all historical interactions with single-choice problemType. |
| `answer_type_choose_1_correct_rate` | Within historical single-choice interactions: first-response correctness fraction. |
| `answer_type_choose_1_attempt_count_mean` | Within historical single-choice interactions: mean reconstructed attempts. |
| `answer_type_choose_1_first_response_time_mean_seconds` | Within historical single-choice interactions: mean first-response time in seconds. |
| `answer_type_choose_1_first_response_time_total_seconds` | Within historical single-choice interactions: total first-response time in seconds. |
| `answer_type_fill_in_1_rate` | Fraction of all historical interactions with fill-in problemType. |
| `answer_type_fill_in_1_correct_rate` | Within historical fill-in interactions: first-response correctness fraction. |
| `answer_type_fill_in_1_attempt_count_mean` | Within historical fill-in interactions: mean reconstructed attempts. |
| `answer_type_fill_in_1_first_response_time_mean_seconds` | Within historical fill-in interactions: mean first-response time in seconds. |
| `answer_type_fill_in_1_first_response_time_total_seconds` | Within historical fill-in interactions: total first-response time in seconds. |
| `previous_problem_correct` | First-response correctness of the last historical interaction in completion order. |
| `previous_problem_attempt_count` | Reconstructed attempt count of the last historical interaction in completion order. |
| `previous_problem_first_response_time_seconds` | First-response time of the last historical interaction in completion order, in seconds. |
| `recent_5_correct_rate` | Across up to five most recent historical interactions in completion order: first-response correctness fraction. |
| `recent_5_attempt_count_mean` | Across up to five most recent historical interactions in completion order: mean reconstructed attempts. |
| `recent_5_first_response_time_mean_seconds` | Across up to five most recent historical interactions in completion order: mean first-response time in seconds. |
| `recent_5_hint_use_rate` | Across up to five most recent historical interactions in completion order: fraction with reconstructed hint_count > 0. |
| `selected_skill_prior_interactions` | Number of historical interactions matching the selected target’s stripped skill label exactly. |
| `selected_skill_correct_rate` | Within history matching the target’s stripped skill label exactly: first-response correctness fraction. |
| `selected_skill_attempt_count_mean` | Within history matching the target’s stripped skill label exactly: mean reconstructed attempts. |
| `selected_skill_first_response_time_mean_seconds` | Within history matching the target’s stripped skill label exactly: mean first-response time in seconds. |
| `selected_skill_hint_use_rate` | Within history matching the target’s stripped skill label exactly: fraction with reconstructed hint_count > 0. |
| `attempt_count_median` | Median of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_p75` | 75th percentile of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_iqr` | Interquartile range (P75 − P25) of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `attempt_count_above_p90_rate` | Fraction strictly above the full-history 90th percentile of attempts per historical interaction, reconstructed as max(attemptCount) across its actions. |
| `first_response_time_median_seconds` | Median of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_p75_seconds` | 75th percentile of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_iqr_seconds` | Interquartile range (P75 − P25) of first-response time (timeTaken) per historical interaction, in seconds. |
| `first_response_time_above_p90_rate` | Fraction strictly above the full-history 90th percentile of first-response time (timeTaken) per historical interaction, in seconds. |
| `recent_5_observation_count` | Number of interactions in the most recent historical window: min(5, history size). |
| `recent_5_attempt_count_median` | Across up to five most recent historical interactions in completion order: median reconstructed attempts. |
| `recent_5_attempt_count_p90` | Across up to five most recent historical interactions in completion order: 90th percentile of reconstructed attempts. |
| `recent_5_first_response_time_median_seconds` | Across up to five most recent historical interactions in completion order: median first-response time in seconds. |
| `recent_5_first_response_time_p90_seconds` | Across up to five most recent historical interactions in completion order: 90th percentile of first-response time in seconds. |
| `selected_skill_has_history` | Indicator that at least one historical interaction matches the selected target’s skill label. |
| `recent_10_correct_trend` | OLS slope of correct against interaction index 0…n−1 over up to ten latest historical interactions in completion order. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(reconstructed attempt count) against interaction index 0…n−1 over up to ten latest historical interactions in completion order. |
| `recent_10_log1p_first_response_time_seconds_trend` | OLS slope of log1p(first-response time in seconds) against interaction index 0…n−1 over up to ten latest historical interactions in completion order. |
| `recent_10_hint_use_trend` | OLS slope of the hint_count > 0 indicator against interaction index 0…n−1 over up to ten latest historical interactions in completion order. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_first_correct` | Binary | First-response correctness (correct) of the final valid interaction selected by maximum startTime. |
| `target_regression_first_response_time` | Regression | First-response timeTaken of the selected target interaction, finite and strictly positive. In seconds. |

## ASSISTments 2012–2013 — SB

Dataset label: `assistments2012_sb`. 82 inputs, 3 targets.

Input location: `datasets/assistments2012-2013/prepared/assistments2012_sb`.

One row per student, predicting the reserved final Skill Builder assignment from earlier assignments whose end precedes its start. General time summaries include scaffolding, correctness, attempts and hints use original problems. Selected-assignment context is retrospectively extracted. Affect inputs are external-detector predictions, not observed states.

### Inputs

| Column | Description |
| --- | --- |
| `original_problem_count` | Number of historical rows with original = 1. |
| `ended_assignment_count` | Number of earlier SB assignments that ended before the target assignment started. |
| `scaffolding_problem_rate` | Fraction of historical interaction rows with original = 0 (scaffolding). |
| `interactions` | Number of all historical interaction rows, including scaffolding. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `correct_rate` | Correct-response fraction among historical original-problem rows. |
| `attempt_count_mean` | Mean of attempt counts across historical original-problem rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical original-problem rows. |
| `attempt_count_total` | Sum of attempt counts across historical original-problem rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical original-problem rows. |
| `hint_count_mean` | Mean hint count among historical original-problem rows. |
| `hint_count_total` | Total hint count across historical original-problem rows. |
| `hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0. |
| `bottom_hint_rate` | Fraction of historical original-problem rows with bottom_hint = 1 (final hint). |
| `first_action_attempt_rate` | Fraction of historical original-problem rows with an attempt as the first action (first_action = 0). |
| `first_action_hint_rate` | Fraction of historical original-problem rows with a hint as the first action (first_action = 1). |
| `first_action_scaffolding_rate` | Fraction of historical original-problem rows with scaffolding as the first action (first_action = 2). |
| `overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_std_seconds` | Standard deviation of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_p90_seconds` | 90th percentile of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `answer_type_algebra_rate` | Fraction of historical original-problem rows with algebra response format. |
| `answer_type_algebra_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical original-problem rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical original-problem rows, restricted to algebra response format. |
| `answer_type_choose_1_rate` | Fraction of historical original-problem rows with single-choice response format. |
| `answer_type_choose_1_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical original-problem rows, restricted to single-choice response format. |
| `answer_type_choose_n_rate` | Fraction of historical original-problem rows with multiple-selection response format. |
| `answer_type_choose_n_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical original-problem rows, restricted to multiple-selection response format. |
| `answer_type_fill_in_1_rate` | Fraction of historical original-problem rows with single-field fill-in response format. |
| `answer_type_fill_in_1_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical original-problem rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical original-problem rows, restricted to single-field fill-in response format. |
| `mastery_rate` | Fraction of historical assignments containing any run of three consecutive correct original problems. |
| `assignment_problem_count_mean` | Mean of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_problem_count_std` | Standard deviation of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_problem_count_p90` | 90th percentile of distinct problems per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_mean` | Mean of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_std` | Standard deviation of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `assignment_time_p90` | 90th percentile of summed recorded interaction work time per assignment, including scaffolding across ended historical assignments. |
| `previous_assignment_mastered` | Any-three mastery indicator of the most recently ended historical assignment. |
| `previous_assignment_problem_count` | Distinct problem_id count, including scaffolding, in the most recently ended historical assignment. |
| `previous_assignment_time` | Sum of recorded interaction work time in the most recently ended historical assignment, including scaffolding. |
| `selected_assignment_answer_type_algebra_present` | Indicator that any retained target-assignment row has algebra response format. |
| `selected_assignment_answer_type_choose_1_present` | Indicator that any retained target-assignment row has single-choice response format. |
| `selected_assignment_answer_type_choose_n_present` | Indicator that any retained target-assignment row has multiple-selection response format. |
| `selected_assignment_answer_type_fill_in_1_present` | Indicator that any retained target-assignment row has single-field fill-in response format. |
| `attempt_count_median` | Median of attempt counts across historical original-problem rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical original-problem rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical original-problem rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical original-problem rows. |
| `overlap_time_median_seconds` | Median of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_p75_seconds` | 75th percentile of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_iqr_seconds` | Interquartile range (P75 minus P25) of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `overlap_time_above_p90_rate` | Fraction strictly above the same history’s P90 of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical original-problem rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding, using only the up to five latest all historical interaction rows, including scaffolding. |
| `recent_5_hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical original-problem rows, using only the up to five latest historical original-problem rows. |
| `recent_5_overlap_time_median_seconds` | Median of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding, using only the up to five latest all historical interaction rows, including scaffolding. |
| `recent_5_overlap_time_p90_seconds` | 90th percentile of interaction duration end_time minus start_time across all historical interaction rows, including scaffolding, using only the up to five latest all historical interaction rows, including scaffolding. |
| `selected_skill_prior_interactions` | Number of earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_has_history` | Indicator that there is at least one of the earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_correct_rate` | Correct-response fraction among historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_attempt_count_mean` | Mean of attempt counts across historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical original-problem rows, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `selected_skill_hint_use_rate` | Fraction of historical original-problem rows with hint_count > 0, restricted to earlier original rows whose skill_id matches the target assignment’s observed skill set. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical original-problem rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical original-problem rows. |
| `recent_10_log1p_overlap_time_seconds_trend` | OLS slope of log1p(interaction time in seconds) against chronological row position in up to ten latest all historical interaction rows, including scaffolding. |
| `recent_10_hint_use_trend` | OLS slope of hint-use indicator (hint_count > 0) against chronological row position in up to ten latest historical original-problem rows. |
| `predicted_frustration_mean` | Historical mean of externally predicted frustration, detector output. |
| `predicted_confusion_mean` | Historical mean of externally predicted confusion, detector output. |
| `predicted_concentration_mean` | Historical mean of externally predicted concentration, detector output. |
| `predicted_boredom_mean` | Historical mean of externally predicted boredom, detector output. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_mastered` | Binary | Any run of three consecutive correct original problems within the target assignment. |
| `target_regression_problems` | Regression | Distinct problem_id count across all retained target-assignment rows, including scaffolding. |
| `target_regression_work_time` | Regression | Sum of recorded work time over all retained target-assignment interactions, including scaffolding. In seconds. |

## ASSISTments 2012–2013 — NSB

Dataset label: `assistments2012_nsb`. 89 inputs, 3 targets.

Input location: `datasets/assistments2012-2013/prepared/assistments2012_nsb`.

One row per student, predicting the final retained non-Skill-Builder problem interaction.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `scaffolding_problem_rate` | Fraction of historical interaction rows with original = 0. |
| `original_problem_count` | Number of historical rows with original = 1. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `hint_count_mean` | Mean hint count among historical interaction rows. |
| `hint_count_total` | Total hint count across historical interaction rows. |
| `hint_use_rate` | Fraction of historical interaction rows with hint_count > 0. |
| `bottom_hint_rate` | Fraction of historical interaction rows with bottom_hint = 1 (final hint). |
| `first_action_attempt_rate` | Fraction of historical interaction rows with an attempt as the first action (first_action = 0). |
| `first_action_hint_rate` | Fraction of historical interaction rows with a hint as the first action (first_action = 1). |
| `first_action_scaffolding_rate` | Fraction of historical interaction rows with scaffolding as the first action (first_action = 2). |
| `last_interaction_original` | Original-problem indicator of the reserved target interaction. |
| `last_interaction_tutor_mode_tutor` | Indicator of tutor mode tutor in the reserved target interaction. |
| `last_interaction_tutor_mode_test` | Indicator of tutor mode test in the reserved target interaction. |
| `last_interaction_tutor_mode_pre` | Indicator of tutor mode pre in the reserved target interaction. |
| `last_interaction_tutor_mode_post` | Indicator of tutor mode post in the reserved target interaction. |
| `last_interaction_answer_type_algebra` | Indicator of response format algebra in the reserved target interaction. |
| `last_interaction_answer_type_choose_1` | Indicator of response format single-choice in the reserved target interaction. |
| `last_interaction_answer_type_choose_n` | Indicator of response format multiple-selection in the reserved target interaction. |
| `last_interaction_answer_type_fill_in_1` | Indicator of response format single-field fill-in in the reserved target interaction. |
| `last_interaction_answer_type_other` | Indicator of response format other or missing in the reserved target interaction. |
| `last_interaction_problem_set_type_linear` | Indicator of problem-set type linear in the reserved target interaction. |
| `last_interaction_problem_set_type_random_iterate` | Indicator of problem-set type random_iterate in the reserved target interaction. |
| `last_interaction_problem_set_type_random_child_order` | Indicator of problem-set type random_child_order in the reserved target interaction. |
| `last_interaction_problem_set_type_numeric_limit` | Indicator of problem-set type numeric_limit in the reserved target interaction. |
| `last_interaction_problem_set_type_choose_condition` | Indicator of problem-set type choose_condition in the reserved target interaction. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_std_seconds` | Standard deviation of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_p90_seconds` | 90th percentile of interaction duration end_time minus start_time across historical interaction rows. |
| `answer_type_algebra_rate` | Fraction of historical interaction rows with algebra response format. |
| `answer_type_choose_1_rate` | Fraction of historical interaction rows with single-choice response format. |
| `answer_type_choose_n_rate` | Fraction of historical interaction rows with multiple-selection response format. |
| `answer_type_fill_in_1_rate` | Fraction of historical interaction rows with single-field fill-in response format. |
| `answer_type_algebra_correct_rate` | Correct-response fraction among historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, restricted to algebra response format. |
| `answer_type_algebra_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical interaction rows, restricted to algebra response format. |
| `answer_type_choose_1_correct_rate` | Correct-response fraction among historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_1_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical interaction rows, restricted to single-choice response format. |
| `answer_type_choose_n_correct_rate` | Correct-response fraction among historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_choose_n_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical interaction rows, restricted to multiple-selection response format. |
| `answer_type_fill_in_1_correct_rate` | Correct-response fraction among historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, restricted to single-field fill-in response format. |
| `answer_type_fill_in_1_overlap_time_total_seconds` | Sum of interaction duration end_time minus start_time across historical interaction rows, restricted to single-field fill-in response format. |
| `previous_problem_correct` | Correctness of the immediately preceding historical interaction. |
| `previous_problem_attempt_count` | Attempt count of the immediately preceding historical interaction. |
| `previous_problem_overlap_time_seconds` | Interaction time of the immediately preceding historical interaction, in seconds. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_hint_use_rate` | Fraction of historical interaction rows with hint_count > 0, using only the up to five latest historical interaction rows. |
| `selected_skill_prior_interactions` | Number of earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_correct_rate` | Correct-response fraction among historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_attempt_count_mean` | Mean of attempt counts across historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_overlap_time_mean_seconds` | Mean of interaction duration end_time minus start_time across historical interaction rows, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `selected_skill_hint_use_rate` | Fraction of historical interaction rows with hint_count > 0, restricted to earlier interactions whose skill intersects the target problem’s skills. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `overlap_time_median_seconds` | Median of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_p75_seconds` | 75th percentile of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_iqr_seconds` | Interquartile range (P75 minus P25) of interaction duration end_time minus start_time across historical interaction rows. |
| `overlap_time_above_p90_rate` | Fraction strictly above the same history’s P90 of interaction duration end_time minus start_time across historical interaction rows. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_median_seconds` | Median of interaction duration end_time minus start_time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_overlap_time_p90_seconds` | 90th percentile of interaction duration end_time minus start_time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `selected_skill_has_history` | Indicator that there is at least one of the earlier interactions whose skill intersects the target problem’s skills. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_overlap_time_seconds_trend` | OLS slope of log1p(interaction time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_hint_use_trend` | OLS slope of hint-use indicator (hint_count > 0) against chronological row position in up to ten latest historical interaction rows. |
| `predicted_frustration_mean` | Historical mean of externally predicted frustration, detector output. |
| `predicted_confusion_mean` | Historical mean of externally predicted confusion, detector output. |
| `predicted_concentration_mean` | Historical mean of externally predicted concentration, detector output. |
| `predicted_boredom_mean` | Historical mean of externally predicted boredom, detector output. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_correct` | Binary | Recorded correctness of the reserved final target interaction. |
| `target_regression_overlap_time` | Regression | Target-interaction duration end_time minus start_time in seconds. |
| `target_regression_attempts` | Regression | Recorded attempt_count of the target interaction. |

## ASSISTments Controlled Experiments

Dataset label: `assistments_controlled_experiments`. 66 inputs, 2 targets.

Input location: `datasets/assistments_from_controlled_experiments/prepared/assistments_controlled_experiments`.

One row per student–problem set, with at least three supplied prior assignments. Historical measures are supplied upstream rather than reconstructed by this producer. Grade, inferred gender, state and selected problem-set indicators are context, the latter supports known-content rather than unseen-content prediction.

### Inputs

| Column | Description |
| --- | --- |
| `prior_problem_count` | Supplied number of problems previously worked by the student. |
| `prior_percent_correct` | Supplied prior-history correctness proportion, on the 0–1 scale. |
| `prior_assignments_assigned` | Number of prior Skill Builder assignments assigned to the student’s class section. |
| `prior_assignment_count` | Number of prior Skill Builder assignments attempted by the student in that class section. |
| `prior_percent_completion` | Fraction of the student’s attempted prior Skill Builders that were completed: completed/attempted. |
| `prior_class_percent_completion` | Prior Skill Builder completion proportion at the class-section level. |
| `z_scored_mastery_speed` | Supplied prior mastery speed, based on problems needed for completion and standardized within the class section, a z-score. |
| `prior_homework_assigned` | Number of prior Skill Builders estimated to have been assigned as homework to the class section, homework status is inferred. |
| `prior_homework_count` | Number of prior Skill Builders estimated to have been attempted by the student outside class in that section. |
| `prior_homework_percent_completion` | Fraction of attempted prior homework Skill Builders completed by the student: completed/attempted. |
| `prior_class_homework_percent_completion` | Prior homework Skill Builder completion proportion at the class-section level. |
| `z_scored_hw_mastery_speed` | Supplied prior homework mastery speed, based on problems needed for completion and standardized within the class section, a z-score. |
| `class_grade_5` | One-hot indicator for supplied Class Grade = 5. |
| `class_grade_6` | One-hot indicator for supplied Class Grade = 6. |
| `class_grade_7` | One-hot indicator for supplied Class Grade = 7. |
| `class_grade_8` | One-hot indicator for supplied Class Grade = 8. |
| `class_grade_9` | One-hot indicator for supplied Class Grade = 9. |
| `class_grade_10` | One-hot indicator for supplied Class Grade = 10. |
| `class_grade_11` | One-hot indicator for supplied Class Grade = 11. |
| `class_grade_12` | One-hot indicator for supplied Class Grade = 12. |
| `class_grade_freshman` | One-hot indicator for supplied Class Grade = Freshmen. |
| `class_grade_sophomore` | One-hot indicator for supplied Class Grade = Sophomore. |
| `class_grade_junior` | One-hot indicator for supplied Class Grade = Junior. |
| `class_grade_senior` | One-hot indicator for supplied Class Grade = Senior. |
| `class_grade_unknown` | One-hot indicator for supplied Class Grade = N/A (unknown grade). |
| `guessed_gender_female` | One-hot indicator for supplied Guessed Gender = Female, inferred gender. |
| `guessed_gender_male` | One-hot indicator for supplied Guessed Gender = Male, inferred gender. |
| `guessed_gender_unknown` | One-hot indicator for supplied Guessed Gender = Uknown (native spelling), inferred gender. |
| `state_5` | One-hot indicator for source State ID = 5, categorical geographic code. |
| `state_7` | One-hot indicator for source State ID = 7, categorical geographic code. |
| `state_9` | One-hot indicator for source State ID = 9, categorical geographic code. |
| `state_10` | One-hot indicator for source State ID = 10, categorical geographic code. |
| `state_20` | One-hot indicator for source State ID = 20, categorical geographic code. |
| `state_22` | One-hot indicator for source State ID = 22, categorical geographic code. |
| `state_23` | One-hot indicator for source State ID = 23, categorical geographic code. |
| `state_30` | One-hot indicator for source State ID = 30, categorical geographic code. |
| `state_31` | One-hot indicator for source State ID = 31, categorical geographic code. |
| `state_33` | One-hot indicator for source State ID = 33, categorical geographic code. |
| `state_34` | One-hot indicator for source State ID = 34, categorical geographic code. |
| `state_36` | One-hot indicator for source State ID = 36, categorical geographic code. |
| `state_39` | One-hot indicator for source State ID = 39, categorical geographic code. |
| `state_44` | One-hot indicator for source State ID = 44, categorical geographic code. |
| `state_48` | One-hot indicator for source State ID = 48, categorical geographic code. |
| `state_53` | One-hot indicator for source State ID = 53, categorical geographic code. |
| `problem_set_226210` | One-hot indicator for selected experimental problem_set = 226210, content identity. |
| `problem_set_237447` | One-hot indicator for selected experimental problem_set = 237447, content identity. |
| `problem_set_241501` | One-hot indicator for selected experimental problem_set = 241501, content identity. |
| `problem_set_241622` | One-hot indicator for selected experimental problem_set = 241622, content identity. |
| `problem_set_243393` | One-hot indicator for selected experimental problem_set = 243393, content identity. |
| `problem_set_246482` | One-hot indicator for selected experimental problem_set = 246482, content identity. |
| `problem_set_246627` | One-hot indicator for selected experimental problem_set = 246627, content identity. |
| `problem_set_246647` | One-hot indicator for selected experimental problem_set = 246647, content identity. |
| `problem_set_250476` | One-hot indicator for selected experimental problem_set = 250476, content identity. |
| `problem_set_255116` | One-hot indicator for selected experimental problem_set = 255116, content identity. |
| `problem_set_256017` | One-hot indicator for selected experimental problem_set = 256017, content identity. |
| `problem_set_256027` | One-hot indicator for selected experimental problem_set = 256027, content identity. |
| `problem_set_259379` | One-hot indicator for selected experimental problem_set = 259379, content identity. |
| `problem_set_263015` | One-hot indicator for selected experimental problem_set = 263015, content identity. |
| `problem_set_263052` | One-hot indicator for selected experimental problem_set = 263052, content identity. |
| `problem_set_263057` | One-hot indicator for selected experimental problem_set = 263057, content identity. |
| `problem_set_263109` | One-hot indicator for selected experimental problem_set = 263109, content identity. |
| `problem_set_263115` | One-hot indicator for selected experimental problem_set = 263115, content identity. |
| `problem_set_293151` | One-hot indicator for selected experimental problem_set = 293151, content identity. |
| `problem_set_303899` | One-hot indicator for selected experimental problem_set = 303899, content identity. |
| `problem_set_377658` | One-hot indicator for selected experimental problem_set = 377658, content identity. |
| `problem_set_377938` | One-hot indicator for selected experimental problem_set = 377938, content identity. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_complete` | Binary | Supplied complete indicator for completion/mastery of the experimental Skill Builder at the student–problem-set level. |
| `target_regression_problems` | Regression | Supplied ProblemCount: problems observed until completion or exit, zero is valid. |

## ASSISTments 2017

Dataset label: `assistments2017`. 38 inputs, 2 targets.

Input location: `datasets/assistments2017/prepared/assistments2017`.

One row per student, predicting MCAS score and subsequent college enrollment. Action-derived summaries use actions starting before the local May 13 cutoff, supplied Ave*, NumActions and predicted_* fields retain their source full-log scope. End-derived spans can cross the start-based cutoff. Model-score training provenance is not established.

### Inputs

| Column | Description |
| --- | --- |
| `usage_year_2005_2006` | Fixed school-year indicator: 2005–2006 = 1, 2004–2005 = 0. |
| `AveKnow` | Supplied mean knowledge probability from Bayesian Knowledge Tracing, model-generated. |
| `AveCarelessness` | Supplied mean carelessness score from an external model, model-generated. |
| `AveCorrect` | Supplied full-log fraction of action rows marked correct, hint actions also enter the denominator. |
| `NumActions` | Supplied full-log number of action rows. |
| `predicted_boredom_mean` | Supplied mean external-detector prediction/rescaled score for boredom, model-generated. |
| `predicted_concentration_mean` | Supplied mean external-detector prediction/rescaled score for engaged concentration, model-generated. |
| `predicted_confusion_mean` | Supplied mean external-detector prediction/rescaled score for confusion, model-generated. |
| `predicted_frustration_mean` | Supplied mean external-detector prediction/rescaled score for frustration, model-generated. |
| `predicted_off_task_mean` | Supplied mean external-detector prediction/rescaled score for off-task behavior, model-generated. |
| `predicted_gaming_mean` | Supplied mean external-detector prediction/rescaled score for gaming the system, model-generated. |
| `skill_distinct_count` | Number of distinct nonempty skill values in retained actions, treating composite labels as single values. |
| `problemId_distinct_count` | Number of distinct problem IDs in retained actions, including scaffolding problems. |
| `assignmentId_distinct_count` | Number of distinct assignment IDs in retained actions. |
| `assistmentId_distinct_count` | Number of distinct assistment IDs in retained actions. |
| `timeTaken_mean_seconds` | Mean finite timeTaken across retained actions, in seconds. |
| `timeTaken_median_seconds` | Median finite timeTaken across retained actions, in seconds. |
| `active_days` | Number of distinct UTC start dates among retained actions. |
| `usage_span_days` | (Maximum endTime − minimum startTime)/86400 across retained actions, in days. |
| `original_rate` | Mean of valid binary original values over retained action rows, indicating an original problem, action-weighted. |
| `hint_rate` | Mean of valid binary hint values over retained action rows, indicating a hint action, action-weighted. |
| `scaffold_rate` | Mean of valid binary scaffold values over retained action rows, indicating a scaffolding action, action-weighted. |
| `frIsHelpRequest_rate` | Mean of valid binary frIsHelpRequest values over retained action rows, indicating a help request on the first response, action-weighted. |
| `stlHintUsed_rate` | Mean of valid binary stlHintUsed values over retained action rows, indicating hint use recorded by stlHintUsed, action-weighted. |
| `endsWithScaffolding_rate` | Mean of valid binary endsWithScaffolding values over retained action rows, indicating ending with scaffolding, action-weighted. |
| `endsWithAutoScaffolding_rate` | Mean of valid binary endsWithAutoScaffolding values over retained action rows, indicating ending with automatic scaffolding, action-weighted. |
| `frWorkingInSchool_rate` | Mean of valid binary frWorkingInSchool values over retained action rows, indicating first response occurring at school, action-weighted. |
| `responseIsFillIn_rate` | Mean of valid binary responseIsFillIn values over retained action rows, indicating a fill-in response, action-weighted. |
| `responseIsChosen_rate` | Mean of valid binary responseIsChosen values over retained action rows, indicating a selected-option response, action-weighted. |
| `Ln_mean` | Mean finite Ln across retained actions: BKT-estimated knowledge probability, explicitly model-generated. |
| `problemType_textfield_rate` | Fraction of all retained actions whose normalized problemType indicates text-field questions. |
| `problemType_radio_rate` | Fraction of all retained actions whose normalized problemType indicates single-choice radio questions. |
| `problemType_algebra_field_rate` | Fraction of all retained actions whose normalized problemType indicates algebra-field questions. |
| `problemType_algebra_rate` | Fraction of all retained actions whose normalized problemType indicates the explicit algebra category. |
| `problemType_popup_menu_rate` | Fraction of all retained actions whose normalized problemType indicates popup-menu questions. |
| `problemType_checkbox_rate` | Fraction of all retained actions whose normalized problemType indicates checkbox questions. |
| `problemType_other_rate` | Fraction of all retained actions whose normalized problemType indicates the explicit other category. |
| `problemType_unknown_rate` | Fraction of all retained actions whose normalized problemType indicates missing, unrecognized, or unknown-coded types. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_regression_mcas` | Regression | External Massachusetts MCAS test score for the school year, -999 denotes missing and is excluded from the cohort. |
| `target_binary_enrolled` | Binary | Supplied indicator of subsequent college enrollment: 1 enrolled, 0 not enrolled. |

## ASSISTments 2019–2020 — SB

Dataset label: `assistments2019_sb`. 61 inputs, 3 targets.

Input location: `datasets/assistments2019-2020/prepared/assistments2019_sb`.

One row per student, predicting the final Skill Builder assignment from earlier fully ended Skill Builder assignments. Mastery uses recorded assignment completion, not reconstructed any-three correctness, problem counts are raw problem-log row counts. Target-assignment problem-type indicators are retrospectively extracted.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `assignment_count` | Number of earlier ended SB assignments. |
| `active_day_count` | Number of distinct UTC dates of historical interaction starts. |
| `history_span_days` | Days between the first and last historical interaction start timestamps. |
| `days_since_previous_interaction` | Days from the most recent historical interaction start to the target start. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_std_seconds` | Standard deviation of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_total_seconds` | Sum of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_observed_rate` | Fraction of historical interactions with observed time_on_task. |
| `answer_given_rate` | Fraction of historical interactions with answer_given = 1, missing or nonnumeric values count as no. |
| `problem_type_algebraic_expression_rate` | Fraction of historical interactions with algebraic expression problem type. |
| `problem_type_number_rate` | Fraction of historical interactions with number problem type. |
| `problem_type_ungraded_open_response_rate` | Fraction of historical interactions with ungraded open response problem type. |
| `problem_type_multiple_choice_rate` | Fraction of historical interactions with multiple choice problem type. |
| `problem_type_exact_match_case_sensitive_rate` | Fraction of historical interactions with case-sensitive exact match problem type. |
| `problem_type_exact_match_ignore_case_rate` | Fraction of historical interactions with case-insensitive exact match problem type. |
| `problem_type_check_all_that_apply_rate` | Fraction of historical interactions with check all that apply problem type. |
| `problem_type_exact_fraction_rate` | Fraction of historical interactions with exact fraction problem type. |
| `problem_type_numeric_expression_rate` | Fraction of historical interactions with numeric expression problem type. |
| `problem_type_ordering_rate` | Fraction of historical interactions with ordering problem type. |
| `problem_type_unknown_rate` | Fraction of historical interactions with unknown or missing problem type. |
| `hint_tutoring_available_rate` | Fraction of historical interactions whose problem offers hint tutoring, availability. |
| `explanation_tutoring_available_rate` | Fraction of historical interactions whose problem offers explanation tutoring, availability. |
| `scaffold_tutoring_available_rate` | Fraction of historical interactions whose problem offers scaffold tutoring, availability. |
| `mastery_rate` | Fraction of historical assignments with recorded assignment_completed = 1. |
| `assignment_problem_count_mean` | Mean of raw plogs rows per assignment across ended historical assignments. |
| `assignment_problem_count_std` | Standard deviation of raw plogs rows per assignment across ended historical assignments. |
| `assignment_problem_count_p90` | 90th percentile of raw plogs rows per assignment across ended historical assignments. |
| `previous_assignment_mastered` | Recorded assignment_completed of the most recently ended historical assignment. |
| `previous_assignment_problem_count` | Raw plogs row count of the most recently ended historical assignment. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_p75_seconds` | 75th percentile of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction active time across historical interaction rows with observed time. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_time_on_task_seconds_trend` | OLS slope of log1p(active time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `assignment_time_on_task_mean_seconds` | Mean of native assignment active time across historical assignments with observed time. |
| `assignment_time_on_task_std_seconds` | Standard deviation of native assignment active time across historical assignments with observed time. |
| `assignment_time_on_task_p90_seconds` | 90th percentile of native assignment active time across historical assignments with observed time. |
| `previous_assignment_time_on_task_seconds` | Native time_on_task of the most recently ended historical assignment, in seconds, missing maps to zero. |
| `assignment_time_on_task_observed_rate` | Fraction of historical assignments with finite native time_on_task. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_mastered` | Binary | Recorded completion of the target assignment: alogs.assignment_completed. |
| `target_regression_problems` | Regression | Raw plogs row count for the target assignment log_id, before behavioral-history cleaning. |
| `target_regression_time_on_task` | Regression | Native assignment-level alogs.time_on_task, neither summed plogs time nor elapsed time. In seconds. |

## ASSISTments 2019–2020 — NSB

Dataset label: `assistments2019_nsb`. 54 inputs, 3 targets.

Input location: `datasets/assistments2019-2020/prepared/assistments2019_nsb`.

One row per student, predicting the final retained problem_set interaction. Earlier history can include both Skill Builder and non-Skill-Builder assignments, the two assignment-rate features describe that composition.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `assignment_count` | Number of distinct assignment_id values in the history. |
| `skill_builder_assignment_rate` | Fraction of distinct historical assignments classified as skill_builder, assignment-level denominator. |
| `problem_set_assignment_rate` | Fraction of distinct historical assignments classified as problem_set, assignment-level denominator. |
| `active_day_count` | Number of distinct UTC dates of historical interaction starts. |
| `history_span_days` | Days between the first and last historical interaction start timestamps. |
| `days_since_previous_interaction` | Days from the most recent historical interaction start to the target start. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows. |
| `time_on_task_std_seconds` | Standard deviation of recorded interaction active time across historical interaction rows. |
| `time_on_task_total_seconds` | Sum of recorded interaction active time across historical interaction rows. |
| `time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows. |
| `answer_given_rate` | Fraction of historical interactions with answer_given = 1, missing or nonnumeric values count as no. |
| `problem_type_algebraic_expression_rate` | Fraction of historical interactions with algebraic expression problem type. |
| `problem_type_number_rate` | Fraction of historical interactions with number problem type. |
| `problem_type_ungraded_open_response_rate` | Fraction of historical interactions with ungraded open response problem type. |
| `problem_type_multiple_choice_rate` | Fraction of historical interactions with multiple choice problem type. |
| `problem_type_exact_match_case_sensitive_rate` | Fraction of historical interactions with case-sensitive exact match problem type. |
| `problem_type_exact_match_ignore_case_rate` | Fraction of historical interactions with case-insensitive exact match problem type. |
| `problem_type_check_all_that_apply_rate` | Fraction of historical interactions with check all that apply problem type. |
| `problem_type_exact_fraction_rate` | Fraction of historical interactions with exact fraction problem type. |
| `problem_type_numeric_expression_rate` | Fraction of historical interactions with numeric expression problem type. |
| `problem_type_ordering_rate` | Fraction of historical interactions with ordering problem type. |
| `problem_type_unknown_rate` | Fraction of historical interactions with unknown or missing problem type. |
| `hint_tutoring_available_rate` | Fraction of historical interactions whose problem offers hint tutoring, availability. |
| `explanation_tutoring_available_rate` | Fraction of historical interactions whose problem offers explanation tutoring, availability. |
| `scaffold_tutoring_available_rate` | Fraction of historical interactions whose problem offers scaffold tutoring, availability. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows. |
| `time_on_task_p75_seconds` | 75th percentile of recorded interaction active time across historical interaction rows. |
| `time_on_task_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction active time across historical interaction rows. |
| `time_on_task_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction active time across historical interaction rows. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_time_on_task_seconds_trend` | OLS slope of log1p(active time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `previous_problem_correct` | Correctness of the immediately preceding historical interaction. |
| `previous_problem_attempt_count` | Attempt count of the immediately preceding historical interaction. |
| `previous_problem_time_on_task_seconds` | Native time_on_task of the immediately preceding historical interaction, in seconds. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_correct` | Binary | Recorded correctness of the reserved final target interaction. |
| `target_regression_time_on_task` | Regression | Native interaction-level plogs.time_on_task of the reserved target. In seconds. |
| `target_regression_attempts` | Regression | Recorded attempt_count of the target interaction. |

## ASSISTments 2020–2021, aug_to_oct — SB

Dataset label: `assistments2021_sb`. 61 inputs, 3 targets.

Input location: `datasets/assistments2020-2021/prepared/assistments2021_sb`.

Same assignment-level task as the 2019–2020 SB branch, restricted to the source-defined aug_to_oct block of 2020–2021, not the full school year. Mastery is recorded completion, and problem counts are raw problem-log row counts. Target-assignment context is retrospectively extracted.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `assignment_count` | Number of earlier ended SB assignments. |
| `active_day_count` | Number of distinct UTC dates of historical interaction starts. |
| `history_span_days` | Days between the first and last historical interaction start timestamps. |
| `days_since_previous_interaction` | Days from the most recent historical interaction start to the target start. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_std_seconds` | Standard deviation of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_total_seconds` | Sum of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_observed_rate` | Fraction of historical interactions with observed time_on_task. |
| `answer_given_rate` | Fraction of historical interactions with answer_given = 1, missing or nonnumeric values count as no. |
| `problem_type_algebraic_expression_rate` | Fraction of historical interactions with algebraic expression problem type. |
| `problem_type_number_rate` | Fraction of historical interactions with number problem type. |
| `problem_type_ungraded_open_response_rate` | Fraction of historical interactions with ungraded open response problem type. |
| `problem_type_multiple_choice_rate` | Fraction of historical interactions with multiple choice problem type. |
| `problem_type_exact_match_case_sensitive_rate` | Fraction of historical interactions with case-sensitive exact match problem type. |
| `problem_type_exact_match_ignore_case_rate` | Fraction of historical interactions with case-insensitive exact match problem type. |
| `problem_type_check_all_that_apply_rate` | Fraction of historical interactions with check all that apply problem type. |
| `problem_type_exact_fraction_rate` | Fraction of historical interactions with exact fraction problem type. |
| `problem_type_numeric_expression_rate` | Fraction of historical interactions with numeric expression problem type. |
| `problem_type_ordering_rate` | Fraction of historical interactions with ordering problem type. |
| `problem_type_unknown_rate` | Fraction of historical interactions with unknown or missing problem type. |
| `hint_tutoring_available_rate` | Fraction of historical interactions whose problem offers hint tutoring, availability. |
| `explanation_tutoring_available_rate` | Fraction of historical interactions whose problem offers explanation tutoring, availability. |
| `scaffold_tutoring_available_rate` | Fraction of historical interactions whose problem offers scaffold tutoring, availability. |
| `mastery_rate` | Fraction of historical assignments with recorded assignment_completed = 1. |
| `assignment_problem_count_mean` | Mean of raw plogs rows per assignment across ended historical assignments. |
| `assignment_problem_count_std` | Standard deviation of raw plogs rows per assignment across ended historical assignments. |
| `assignment_problem_count_p90` | 90th percentile of raw plogs rows per assignment across ended historical assignments. |
| `previous_assignment_mastered` | Recorded assignment_completed of the most recently ended historical assignment. |
| `previous_assignment_problem_count` | Raw plogs row count of the most recently ended historical assignment. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_p75_seconds` | 75th percentile of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction active time across historical interaction rows with observed time. |
| `time_on_task_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction active time across historical interaction rows with observed time. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows with observed time, using only the up to five latest historical interaction rows. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_time_on_task_seconds_trend` | OLS slope of log1p(active time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `assignment_time_on_task_mean_seconds` | Mean of native assignment active time across historical assignments with observed time. |
| `assignment_time_on_task_std_seconds` | Standard deviation of native assignment active time across historical assignments with observed time. |
| `assignment_time_on_task_p90_seconds` | 90th percentile of native assignment active time across historical assignments with observed time. |
| `previous_assignment_time_on_task_seconds` | Native time_on_task of the most recently ended historical assignment, in seconds, missing maps to zero. |
| `assignment_time_on_task_observed_rate` | Fraction of historical assignments with finite native time_on_task. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_mastered` | Binary | Recorded completion of the target assignment: alogs.assignment_completed. |
| `target_regression_problems` | Regression | Raw plogs row count for the target assignment log_id, before behavioral-history cleaning. |
| `target_regression_time_on_task` | Regression | Native assignment-level alogs.time_on_task, neither summed plogs time nor elapsed time. In seconds. |

## ASSISTments 2020–2021, aug_to_oct — NSB

Dataset label: `assistments2021_nsb`. 54 inputs, 3 targets.

Input location: `datasets/assistments2020-2021/prepared/assistments2021_nsb`.

Same interaction-level task as the 2019–2020 NSB branch, restricted to aug_to_oct. Earlier block-local history can contain both Skill Builder and non-Skill-Builder assignments.

### Inputs

| Column | Description |
| --- | --- |
| `interactions` | Number of historical interaction rows. |
| `repeated_problem_rate` | Repetition fraction: 1 minus distinct historical problem_id count divided by historical row count. |
| `assignment_count` | Number of distinct assignment_id values in the history. |
| `skill_builder_assignment_rate` | Fraction of distinct historical assignments classified as skill_builder, assignment-level denominator. |
| `problem_set_assignment_rate` | Fraction of distinct historical assignments classified as problem_set, assignment-level denominator. |
| `active_day_count` | Number of distinct UTC dates of historical interaction starts. |
| `history_span_days` | Days between the first and last historical interaction start timestamps. |
| `days_since_previous_interaction` | Days from the most recent historical interaction start to the target start. |
| `correct_rate` | Correct-response fraction among historical interaction rows. |
| `attempt_count_mean` | Mean of attempt counts across historical interaction rows. |
| `attempt_count_std` | Standard deviation of attempt counts across historical interaction rows. |
| `attempt_count_total` | Sum of attempt counts across historical interaction rows. |
| `attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows. |
| `time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows. |
| `time_on_task_std_seconds` | Standard deviation of recorded interaction active time across historical interaction rows. |
| `time_on_task_total_seconds` | Sum of recorded interaction active time across historical interaction rows. |
| `time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows. |
| `answer_given_rate` | Fraction of historical interactions with answer_given = 1, missing or nonnumeric values count as no. |
| `problem_type_algebraic_expression_rate` | Fraction of historical interactions with algebraic expression problem type. |
| `problem_type_number_rate` | Fraction of historical interactions with number problem type. |
| `problem_type_ungraded_open_response_rate` | Fraction of historical interactions with ungraded open response problem type. |
| `problem_type_multiple_choice_rate` | Fraction of historical interactions with multiple choice problem type. |
| `problem_type_exact_match_case_sensitive_rate` | Fraction of historical interactions with case-sensitive exact match problem type. |
| `problem_type_exact_match_ignore_case_rate` | Fraction of historical interactions with case-insensitive exact match problem type. |
| `problem_type_check_all_that_apply_rate` | Fraction of historical interactions with check all that apply problem type. |
| `problem_type_exact_fraction_rate` | Fraction of historical interactions with exact fraction problem type. |
| `problem_type_numeric_expression_rate` | Fraction of historical interactions with numeric expression problem type. |
| `problem_type_ordering_rate` | Fraction of historical interactions with ordering problem type. |
| `problem_type_unknown_rate` | Fraction of historical interactions with unknown or missing problem type. |
| `hint_tutoring_available_rate` | Fraction of historical interactions whose problem offers hint tutoring, availability. |
| `explanation_tutoring_available_rate` | Fraction of historical interactions whose problem offers explanation tutoring, availability. |
| `scaffold_tutoring_available_rate` | Fraction of historical interactions whose problem offers scaffold tutoring, availability. |
| `attempt_count_median` | Median of attempt counts across historical interaction rows. |
| `attempt_count_p75` | 75th percentile of attempt counts across historical interaction rows. |
| `attempt_count_iqr` | Interquartile range (P75 minus P25) of attempt counts across historical interaction rows. |
| `attempt_count_above_p90_rate` | Fraction strictly above the same history’s P90 of attempt counts across historical interaction rows. |
| `time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows. |
| `time_on_task_p75_seconds` | 75th percentile of recorded interaction active time across historical interaction rows. |
| `time_on_task_iqr_seconds` | Interquartile range (P75 minus P25) of recorded interaction active time across historical interaction rows. |
| `time_on_task_above_p90_rate` | Fraction strictly above the same history’s P90 of recorded interaction active time across historical interaction rows. |
| `recent_5_observation_count` | Number of rows in the window of up to five latest historical interaction rows. |
| `recent_5_correct_rate` | Correct-response fraction among historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_mean` | Mean of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_mean_seconds` | Mean of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_median` | Median of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_attempt_count_p90` | 90th percentile of attempt counts across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_median_seconds` | Median of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_5_time_on_task_p90_seconds` | 90th percentile of recorded interaction active time across historical interaction rows, using only the up to five latest historical interaction rows. |
| `recent_10_correct_trend` | OLS slope of correctness (0/1) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_attempt_count_trend` | OLS slope of log1p(attempt count) against chronological row position in up to ten latest historical interaction rows. |
| `recent_10_log1p_time_on_task_seconds_trend` | OLS slope of log1p(active time in seconds) against chronological row position in up to ten latest historical interaction rows. |
| `previous_problem_correct` | Correctness of the immediately preceding historical interaction. |
| `previous_problem_attempt_count` | Attempt count of the immediately preceding historical interaction. |
| `previous_problem_time_on_task_seconds` | Native time_on_task of the immediately preceding historical interaction, in seconds. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_correct` | Binary | Recorded correctness of the reserved final target interaction. |
| `target_regression_time_on_task` | Regression | Native interaction-level plogs.time_on_task of the reserved target. In seconds. |
| `target_regression_attempts` | Regression | Recorded attempt_count of the target interaction. |

## NedBox

Dataset label: `nedbox`. 51 inputs, 2 targets.

Input location: `datasets/nedbox/prepared/nedbox`.

One row per user, targeting the selected latest teaser (content item containing publications and exercises). History precedes that teaser’s first session, exact duplicates are removed and latest exercise occurrences are retained separately within history and target. Cold-start users are allowed. Dropout means observed released-catalog-relative incompleteness, not permanent abandonment. Duration and gap units are inherited without a verified physical conversion.

### Inputs

| Column | Description |
| --- | --- |
| `teasers` | Number of distinct identifiable historical teasers. |
| `sessions` | Number of distinct retained historical sessions. |
| `publications` | Number of distinct historical publications, including those without a known teaser. |
| `exercises` | Number of distinct retained historical exercises. |
| `answer_rate` | Fraction of historical interaction rows marked as answered. |
| `incomplete_rate` | Fraction of started historical teasers that remain incomplete relative to the released catalog. |
| `no_response_rate` | Fraction of identifiable historical teasers without any answered exercise. |
| `score_ratio` | Historical score weighted by the maximum scores of valid answered exercises. |
| `score_teaser_mean` | Mean normalized score across historical teasers with valid answered exercises. |
| `score_teaser_std` | Population standard deviation of normalized historical teaser scores. |
| `score_last` | Normalized score of the latest eligible historical teaser, ordered by its first session. |
| `duration_total` | Sum of recorded durations across retained historical interaction rows. |
| `duration_median` | Median recorded duration per historical interaction row. |
| `duration_p90` | 90th percentile of recorded durations per historical interaction row. |
| `session_duration_mean` | Mean of summed recorded interaction durations per historical session. |
| `session_duration_last` | Sum of recorded interaction durations in the latest retained historical session. |
| `gap_mean` | Mean of recorded historical inter-session gaps, using one row per session and excluding session number 1. |
| `gap_median` | Median of recorded historical inter-session gaps, using one row per session and excluding session number 1. |
| `gap_p90` | 90th percentile of recorded historical inter-session gaps, using one row per session and excluding session number 1. |
| `gap_last` | Latest value of recorded historical inter-session gaps, using one row per session and excluding session number 1. |
| `exercise_type_count` | Number of distinct exercise types in the historical interactions. |
| `theme_count` | Number of distinct themes across identifiable historical teasers. |
| `video_rate` | Fraction of distinct identifiable historical teasers marked as video. |
| `context_known_rate` | Fraction of retained historical sessions with an identified teaser after joining the catalog. |
| `difficulty_alpha_rate` | Fraction of known-context historical sessions assigned difficulty alpha. |
| `difficulty_n1_rate` | Fraction of known-context historical sessions assigned difficulty n1. |
| `difficulty_n2_rate` | Fraction of known-context historical sessions assigned difficulty n2. |
| `difficulty_n3_rate` | Fraction of known-context historical sessions assigned difficulty n3. |
| `skill_kijken_rate` | Fraction of known-context historical sessions assigned visual comprehension (kijken). |
| `skill_luisteren_rate` | Fraction of known-context historical sessions assigned listening (luisteren). |
| `skill_lezen_rate` | Fraction of known-context historical sessions assigned reading (lezen). |
| `skill_woordenschat_rate` | Fraction of known-context historical sessions assigned vocabulary (woordenschat). |
| `skill_grammatica_rate` | Fraction of known-context historical sessions assigned grammar (grammatica). |
| `skill_schrijven_rate` | Fraction of known-context historical sessions assigned writing (schrijven). |
| `skill_spreken_rate` | Fraction of known-context historical sessions assigned speaking (spreken). |
| `selected_is_video` | Video indicator for the selected teaser, known before its outcomes. |
| `selected_theme_count` | Number of theme entries in the selected teaser, including repeated entries. |
| `selected_publications` | Number of distinct cataloged publications belonging to the selected teaser. |
| `selected_exercises` | Number of cataloged exercise rows belonging to the selected teaser. |
| `selected_max_score` | Sum of cataloged maximum exercise scores in the selected teaser. |
| `selected_difficulty_alpha_rate` | Fraction of cataloged exercises in the selected teaser assigned difficulty alpha. |
| `selected_difficulty_n1_rate` | Fraction of cataloged exercises in the selected teaser assigned difficulty n1. |
| `selected_difficulty_n2_rate` | Fraction of cataloged exercises in the selected teaser assigned difficulty n2. |
| `selected_difficulty_n3_rate` | Fraction of cataloged exercises in the selected teaser assigned difficulty n3. |
| `selected_skill_kijken_rate` | Fraction of cataloged exercises in the selected teaser assigned visual comprehension (kijken). |
| `selected_skill_luisteren_rate` | Fraction of cataloged exercises in the selected teaser assigned listening (luisteren). |
| `selected_skill_lezen_rate` | Fraction of cataloged exercises in the selected teaser assigned reading (lezen). |
| `selected_skill_woordenschat_rate` | Fraction of cataloged exercises in the selected teaser assigned vocabulary (woordenschat). |
| `selected_skill_grammatica_rate` | Fraction of cataloged exercises in the selected teaser assigned grammar (grammatica). |
| `selected_skill_schrijven_rate` | Fraction of cataloged exercises in the selected teaser assigned writing (schrijven). |
| `selected_skill_spreken_rate` | Fraction of cataloged exercises in the selected teaser assigned speaking (spreken). |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_dropout` | Binary | Started-but-incomplete target teaser: at least one answered difficulty remains incomplete relative to its catalog. |
| `target_regression_score` | Regression | Target teaser score: achieved points divided by maximum points over valid answered exercises only. |

## Auction Verification

Dataset label: `auction_preprocessed`. 7 inputs, 2 targets.

Input location: `datasets/auction_preprocessed/auction_preprocessed.csv`.

One row per formal-verification run of a simultaneous multi-round auction. Predict whether the queried outcome is possible and how long verification takes, this is not a dataset of observed sale prices.

### Inputs

| Column | Description |
| --- | --- |
| `process.b1.capacity` | Maximum number of products bidder 1 may win. |
| `process.b2.capacity` | Maximum number of products bidder 2 may win. |
| `process.b3.capacity` | Maximum number of products bidder 3 may win. |
| `process.b4.capacity` | Maximum number of products bidder 4 may win. |
| `property.price` | Price whose attainability is being verified. |
| `property.product` | Product whose outcome is being verified. |
| `property.winner` | Bidder being checked as the winner, 0 means that only the price is checked. |

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_result` | Binary | Whether the queried auction outcome is possible according to formal model checking. |
| `target_regression_time` | Regression | Runtime of the formal verification procedure, in the source's time unit. |

## SPR X-ray

Dataset label: `spr_xray_manifest`. 768 inputs, 2 targets.

Input location: `datasets/spr xray/prepared/spr_xray_manifest`.

One row per radiograph. Inputs are 768 extracted embeddings from the RAD-DINO vision transformer model.

### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_age` | Regression | Recorded age associated with the radiograph image, in years. |
| `target_binary_gender` | Binary | Recorded binary gender label, preserved without recoding by the loader. |

## HIGGS 50k

Dataset label: `higgs_50k`. 21 inputs, 8 targets.

Input location: `datasets/higgs/higgs_50k.csv`.

One row per simulated particle event in the fixed 50k cohort. Four encoded categorical and seventeen numerical inputs are preserved from the upstream MultiTab HDF5. Their generic names do not establish a physical-variable mapping. Upstream global imputation/encoding/scaling is inherited, kinematic-target normalization and physical units are not established here.


### Targets

| Column | Task | Description |
| --- | --- | --- |
| `target_binary_Target` | Binary | Original binary Target label preserved from the MultiTab HDF5. |
| `target_regression_m_bb` | Regression | Upstream-prepared high-level kinematic quantity m_bb preserved as a regression target. |
| `target_regression_m_jj` | Regression | Upstream-prepared high-level kinematic quantity m_jj preserved as a regression target. |
| `target_regression_m_jjj` | Regression | Upstream-prepared high-level kinematic quantity m_jjj preserved as a regression target. |
| `target_regression_m_jlv` | Regression | Upstream-prepared high-level kinematic quantity m_jlv preserved as a regression target. |
| `target_regression_m_lv` | Regression | Upstream-prepared high-level kinematic quantity m_lv preserved as a regression target. |
| `target_regression_m_wbb` | Regression | Upstream-prepared high-level kinematic quantity m_wbb preserved as a regression target. |
| `target_regression_m_wwbb` | Regression | Upstream-prepared high-level kinematic quantity m_wwbb preserved as a regression target. |

## Hydraulic:

The 17 sensor views describe the same 2,205 load cycles. Each row summarizes one 60-second cycle from one sensor.

### Shared inputs

| Column | Description |
| --- | --- |
| `value__mean` | Mean sensor value within the cycle. |
| `value__median` | Median sensor value within the cycle. |
| `value__standard_deviation` | Standard deviation of sensor values. |
| `value__variance` | Variance of sensor values. |
| `value__skewness` | Asymmetry of the sensor-value distribution. |
| `value__kurtosis` | Kurtosis of the sensor-value distribution. |
| `value__maximum` | Maximum sensor value. |
| `value__minimum` | Minimum sensor value. |
| `value__mean_abs_change` | Mean absolute difference between consecutive readings. |
| `value__mean_change` | Mean signed difference between consecutive readings. |
| `value__absolute_sum_of_changes` | Sum of absolute differences between consecutive readings. |
| `value__longest_strike_above_mean` | Longest consecutive run of readings above the cycle mean. |
| `value__longest_strike_below_mean` | Longest consecutive run of readings below the cycle mean. |
| `value__count_above_mean` | Number of readings above the cycle mean. |
| `value__count_below_mean` | Number of readings below the cycle mean. |
| `value__first_location_of_maximum` | First maximum's position divided by the number of readings. |
| `value__first_location_of_minimum` | First minimum's position divided by the number of readings. |
| `value__sample_entropy` | Sample entropy, summarizing temporal irregularity. |
| `value__binned_entropy__max_bins_10` | Entropy of the sensor-value histogram using ten bins. |
| `value__linear_trend__attr_"slope"` | Slope of the linear trend fitted over reading positions. |
| `value__linear_trend__attr_"intercept"` | Intercept of the linear trend fitted over reading positions. |
| `value__autocorrelation__lag_1` | Autocorrelation at a lag of one reading. |
| `value__autocorrelation__lag_2` | Autocorrelation at a lag of two readings. |
| `value__fft_coefficient__attr_"abs"__coeff_0` | Magnitude of the zeroth discrete Fourier coefficient (DC component). |
| `value__fft_coefficient__attr_"abs"__coeff_1` | Magnitude of the first discrete Fourier coefficient. |

### Shared targets

| Column | Task | Description |
| --- | --- | --- |
| `target_cooler_condition_regression` | Regression | Cooler efficiency level (%): 3, 20 or 100. |
| `target_valve_condition_regression` | Regression | Valve condition (%): 73, 80, 90 or 100. |
| `target_internal_pump_leakage_regression` | Regression | Pump leakage severity: 0 = none, 1 = weak, 2 = severe. |
| `target_hydraulic_bar_regression` | Regression | Hydraulic accumulator pressure (bar): 90, 100, 115 or 130. |
| `target_stable_flag_binary` | Binary | 0 = stable conditions, 1 = steady-state conditions may not yet have been reached. |

## Preprocessing from original sources

First, install the files described below and run the corresponding preprocessing command. Paths are relative to the repository root. Follow each provider's access and usage terms.

### Preprocessing workflow

Without `--work-root`, `preprocess` runs the source preprocessor and creates the registered processed input. For example, after installing the WHAS1 source:

```bash
.venv/bin/python reproduction/pipeline.py preprocess --label whas1_preprocessed
```

Then run `prepare` and `train` as shown in the README. Existing processed outputs are protected against overwriting.

With `--work-root`, `preprocess` combines source reconstruction with experiment-input preparation and MTT export:

```bash
.venv/bin/python reproduction/pipeline.py preprocess --label assistments2009_sb --work-root artifacts/assistments2009-sb-reconstructed --workers 1
```

For ASSISTments, this reconstructs the five folds from the registered original sources. Use the resulting work directory with `train`. The work directory stores preparation artifacts, not the original source files. Use a new directory and install the required sources first. Datasets without a registered reconstruction route require their processed input to be installed separately.

If the registered ASSISTments bundle is missing, `prepare` also attempts reconstruction from the registered sources. With an installed bundle, it reuses the existing folds instead.

HIGGS and SPR cohort reconstruction uses the included official selection maps:

```bash
.venv/bin/python reproduction/pipeline.py preprocess --label higgs_50k --input datasets/higgs/train_val_test.h5

.venv/bin/python reproduction/pipeline.py preprocess --label spr_xray_manifest --input '/path/to/spr xray' --output artifacts/spr-raw-selected --device cuda:0 --batch-size 16
```

The SPR `--input` accepts either a full-embedding directory or a Kaggle dataset root containing `train_age.csv`, `train_gender.csv`, and the literal `kaggle/kaggle/train/` subtree. The image route needs only those images and labels plus the included selection CSV. If using `--output` above, install the resulting selected-cohort files at the registered SPR input path before running `prepare`.

For command options, run `.venv/bin/python reproduction/pipeline.py preprocess --help`.

### Source downloads and installation paths

#### SUPPORT

Source: [Vanderbilt/HBiostat](https://hbiostat.org/data/) — [TSV](https://hbiostat.org/data/repo/support.tsv)

Download `support.tsv` unchanged to `datasets/support_withoutmissing_outputs/support.tsv`, then run `.venv/bin/python reproduction/pipeline.py preprocess --label support_withoutmissing_outputs`. The reader accepts the original tab-delimited format without manual CSV conversion. It also accepts `support.csv` and gives that CSV priority when both are present. Preprocessing generates `support_withoutmissing_outputs.csv` and the auxiliary `support_withmissing.csv`.

#### SUPPORT2

Source: [Vanderbilt/HBiostat](https://hbiostat.org/data/) — [CSV ZIP](https://hbiostat.org/data/repo/support2csv.zip)

Extract `support2.csv` to `datasets/support2_withoutmissing_outputs/support2.csv`.

#### WHAS1 (481 records)

Download: [Google Drive](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP).

Extract `whas1_preprocessed.zip`, take `whas1.csv` and place it at `datasets/whas1_preprocessed/raw/whas1.csv` to run preprocessing. The reader also accepts the original `whas.dat` in the same raw directory.

#### WHAS500

Download: [Google Drive](https://drive.google.com/drive/u/3/folders/1-1jRbhQtV8b_8Dpwamz2SkG_xUxzGaKP).

Extract `whas500_preprocessed.zip`, take `whas500.csv` and place it at `datasets/whas500_preprocessed/raw/whas500.csv` to run preprocessing.

#### ASSISTments 2009–2010 SB/NSB

Source: [SB release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010), [NSB release](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/non-skill-builder-data-2009-10)

Install the matching original releases as `datasets/assistments2009-2010/sb.csv` and `nsb.csv`. Original Drive links may require Google access. The corrected/collapsed SB export and the combined Full Data export are not verified substitutes for these benchmark sources.

#### ASSISTments Goldstein P(J), 2011

Source: [Official page](https://sites.google.com/site/assistmentsdata/datasets/2011-goldstein-baker-heffernan) — [WPI ZIP](http://web.cs.wpi.edu/~nth/PublicScienceArchive/assistments-p(j)-data.zip)

Extract `PJ Data (Features Added).csv` and `PJ Data (Actions).csv` to `datasets/assistments_paper:2011_goldstein/`. Preserve spaces, parentheses and the colon in the directory name.

#### ASSISTments 2012–2013

Source: [Official release](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) — [ZIP](https://drive.google.com/file/d/1cU6Ft4R3hLqA7G1rIGArVfelSZvc6RxY/view?usp=sharing)

Extract the CSV with affect predictions and install as `datasets/assistments2012-2013/data.csv`. Preserve the `actions` column. One source serves both SB and NSB.

#### ASSISTments Controlled Experiments

Source: [Study supplement](https://sites.google.com/site/las2016data/data/thison) — [ThisOne.csv](https://drive.google.com/file/d/16zGibMINqvQAef_FkP9f78Y4OlON4NrX/view?usp=drive_web)

Save as `datasets/assistments_from_controlled_experiments/ThisOne.csv`. Use the CSV release, not an arbitrary export of the spreadsheet.

#### ASSISTments 2017

Source: [Official access page](https://sites.google.com/view/assistmentsdatamining/dataset)

Complete the provider's access form and accept its terms. The download link is sent by email. Install as `datasets/assistments2017/Assistments2017.csv`.

#### ASSISTments 2019–2020


Source: [OSF release](https://osf.io/q7zc5/) — [full-school-year ZIP](https://osf.io/download/7sftn/)

Extract the complete archive to `datasets/assistments2019-2020/2019-2020_school_year/`, preserving filenames. The readers need `adets.csv`, `pdets.csv`, `plogs.csv`, `alogs.csv`, and `slogs.csv`. Do not substitute the pre/post-shutdown subsets.

#### ASSISTments 2020–2021

Source: [OSF release](https://osf.io/7cgav/) — [aug_to_oct ZIP](https://osf.io/download/gny98/)

Extract this block to `datasets/assistments2020-2021/aug_to_oct/`, preserving filenames. The readers need `adets.csv`, `pdets.csv`, `plogs.csv`, `alogs.csv`, and `slogs.csv`. Other period blocks are not inputs to this benchmark.

#### Auction Verification

Source: [UCI release](https://archive.ics.uci.edu/dataset/713/auction+verification) — [ZIP](https://archive.ics.uci.edu/static/public/713/auction+verification.zip)

Place the release's data CSV at `datasets/auction_preprocessed/data.csv`, retaining its original header.

#### SPR X-ray

Source: [Kaggle dataset page](https://www.kaggle.com/datasets/felipekitamura/spr-x-ray-age-and-gender-dataset)

Image downloads may require a Kaggle account and accepted competition rules. Raw extraction needs labeled training PNGs in `kaggle/kaggle/train/` plus `train_age.csv` and `train_gender.csv` at the Kaggle dataset root.

#### HIGGS source used by this pipeline

Source: [MultiTab](https://github.com/Armanfard-Lab/MultiTab) — [authors' HDF5 collection](https://huggingface.co/datasets/Dimitri-Sinodinos/multitab/tree/main) — [higgs.h5](https://huggingface.co/datasets/Dimitri-Sinodinos/multitab/resolve/main/higgs.h5?download=true)

Download `higgs.h5`, rename it to `train_val_test.h5`, and place it at `datasets/higgs/train_val_test.h5`. This is an upstream prepared HDF5, not the UCI raw CSV. The HIGGS cohort reconstruction command expects this HDF5 and the repository's fixed selection map.

#### Hydraulic (17 sensor views)

Source: [UCI release](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems) — [ZIP](https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip)

Extract the sensor `*.txt` files and `profile.txt` into `datasets/hydraulic/data/`. Run `preprocess --label <sensor> --work-root <new-directory>` to extract that sensor's tsfresh features and prepare its folds/MTT inputs. Existing processed CSVs are protected against overwriting. CP is written to `datasets/EPS1/CP.csv`. The other views use `datasets/<sensor>/<sensor>.csv`.

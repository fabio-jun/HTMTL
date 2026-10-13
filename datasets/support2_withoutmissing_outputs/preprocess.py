import pandas as pd


def main():
    # outcomes 
    # 1) hospdead = death during hospitalization, binary
    # 2) sfdm2 = the severity of the disfuncionality (check screenshoot), converted from ordinal to numeric
    # 3) charges = total cost of hospital stay, numeric
    # 4) death = death after 5.5 years

    data = pd.read_csv("support2.csv")

    outputs_columns = ["hospdead", "sfdm2", "charges", "death"]
    features_columns = ["age", "sex", "dzgroup", "dzclass", "num.co", "edu", "income", "scoma", 
                "avtisst", "race", "sps", "aps", "diabetes", "dementia", "ca",
                "meanbp", "wblc" ,"hrt" , "resp", "temp" ,"pafi", "alb", 
                "bili", "crea", "sod" ,"ph", "glucose", "bun", "urine", "adlp", "adls", "adlsc"]


    ### convert sfdm2 to numeric
    sfdm2_dict = {
    "no(M2 and SIP pres)" : 1,
    "adl>=4 (>=5 if sur)" : 2,
    "SIP>=30" : 3,
    "Coma or Intub" : 4,
    "<2 mo. follow-up" : 5
    }

    data["sfdm2"] = data["sfdm2"].replace(sfdm2_dict)

    ## remove features with more than 30% missing
    threshold_missing = 0.3
    features_missing = data.isna().sum(axis=0)/data.shape[0] > threshold_missing
    features_to_drop = [i for i,_ in features_missing.items()]
    data = data[features_to_drop]

    outputs_final = data[outputs_columns].reset_index(drop=True)
    features_final = data[features_columns].reset_index(drop=True)

    features_final["sex"] = features_final["sex"].map({"male":0, "female":1})

    features_final = pd.get_dummies(features_final, dtype = int, dummy_na = True)

    dataset_final = pd.concat([features_final, outputs_final], axis = 1)
    dataset_final = dataset_final.rename(columns = {"hospdead": "target_binary_hospdead", "sfdm2" : "target_regression_sfdm2", "charges" : "target_regression_charges", "death" : "target_binary_death"})

    ## only complete cases
    dataset_final_no_missing_outputs = dataset_final.dropna(subset = [c for c in dataset_final.columns if "target" in c])
    dataset_final_no_missing_outputs.columns = ["target_" + c if c in outputs_columns else c for c in dataset_final_no_missing_outputs.columns ]
    dataset_final_no_missing_outputs.to_csv("support2_withoutmissing_outputs.csv", index = False)

    dataset_final.columns = ["target_" + c if c in outputs_columns else c for c in dataset_final.columns ]
    dataset_final.to_csv("support2_withmissing.csv", index = False)


if __name__ == "__main__":
    main()

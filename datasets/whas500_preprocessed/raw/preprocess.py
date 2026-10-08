import pandas as pd


def main():
    data = pd.read_csv("whas500.csv")

    features_columns = ["age",
                "gender",
                "hr",
                "sysbp",
                "diasbp",
                "bmi",
                "cvd",
                "afb",
                "sho",
                "chf",
                "av3",
                "miord",
                "mitype"]

    output_columns = ["los", # length of stay
               "dstat"] # discharged alive (0) or dead (1)
    features = data[features_columns].reset_index(drop = True)
    #print(features["bmi"].replace(",","."))

    features["bmi"] = features["bmi"].str.replace(",",".").astype(float) 
    output = data[output_columns].reset_index(drop = True)
    output.columns = ["target_" + c for c in output.columns]

    dataset_final = pd.concat([features, output], axis = 1)
    dataset_final = dataset_final.rename(columns = {"target_los": "target_regression_los", "target_dstat" : "target_binary_dstat"})

    dataset_final.to_csv("../whas500_preprocessed.csv", index = False)


if __name__ == "__main__":
    main()

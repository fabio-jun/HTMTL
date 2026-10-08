import pandas as pd


def main():
    data = pd.read_csv("whas1.csv")

    features_columns = ["AGE",
                "SEX",
                "CPK",
                "SHO",
                "CHF",
                "MIORD",
                "MITYPE"]

    output_columns = ["LENSTAY",
              "DSTAT"]

    features = data[features_columns].reset_index(drop = True)
    features = pd.get_dummies(features, columns=["MITYPE"], dtype=int)
    output = data[output_columns].reset_index(drop = True)
    output.columns = ["target_" + c for c in output.columns]

    dataset_final = pd.concat([features, output], axis = 1)
    dataset_final = dataset_final.rename(columns = {"target_LENSTAY": "target_regression_LENSTAY", "target_DSTAT" : "target_binary_DSTAT"})

    dataset_final.to_csv("../whas1_preprocessed.csv", index = False)


if __name__ == "__main__":
    main()

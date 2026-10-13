from pathlib import Path
import pandas as pd


SOURCE_COLUMNS = ('ID', 'AGE', 'SEX', 'CPK', 'SHO', 'CHF', 'MIORD', 'MITYPE', 'YEAR', 'YRGRP', 'LENSTAY', 'DSTAT', 'LENFOL', 'FSTAT')


def load_source(directory=Path('.')):
    directory = Path(directory)
    csv_path = directory / 'whas1.csv'
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.read_csv(directory / 'whas.dat', sep=r'\s+', header=None, names=list(SOURCE_COLUMNS), na_values='.')


def main():
    data = load_source()

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

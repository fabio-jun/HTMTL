from pathlib import Path
import pandas as pd


SOURCE_COLUMNS = ('id', 'age', 'gender', 'hr', 'sysbp', 'diasbp', 'bmi', 'cvd', 'afb', 'sho', 'chf', 'av3', 'miord', 'mitype', 'year', 'admitdate', 'disdate', 'fdate', 'los', 'dstat', 'lenfol', 'fstat')


def load_source(directory=Path('.')):
    directory = Path(directory)
    csv_path = directory / 'whas500.csv'
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    return pd.read_csv(directory / 'whas500.dat', sep=r'\s+', header=None, names=list(SOURCE_COLUMNS), na_values='.')


def normalize_bmi(values):
    return values.astype(str).str.replace(',', '.', regex=False).astype(float)


def main():
    data = load_source()

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

    features["bmi"] = normalize_bmi(features["bmi"])
    output = data[output_columns].reset_index(drop = True)
    output.columns = ["target_" + c for c in output.columns]

    dataset_final = pd.concat([features, output], axis = 1)
    dataset_final = dataset_final.rename(columns = {"target_los": "target_regression_los", "target_dstat" : "target_binary_dstat"})

    dataset_final.to_csv("../whas500_preprocessed.csv", index = False)


if __name__ == "__main__":
    main()

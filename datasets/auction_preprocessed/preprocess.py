import pandas as pd


def main():
    data = pd.read_csv("data.csv").astype(float)
    #data.columns = [d.replace("verification.", "target_") for d in data.columns]
    data.rename(columns={"verification.result": "target_binary_result", "verification.time": "target_regression_time"}, inplace=True)
    data.to_csv("auction_preprocessed.csv", index = False)


if __name__ == "__main__":
    main()

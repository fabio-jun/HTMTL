import pandas as pd
from tsfresh import extract_features
#from tsfresh.feature_extraction import MinimalFCParameters

from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

fc_parameters = {
    "mean": None,
    "median": None,
    "standard_deviation": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "maximum": None,
    "minimum": None,
    #"range": None,
    "mean_abs_change": None,
    "mean_change": None,

    "absolute_sum_of_changes": None,
    "longest_strike_above_mean": None,
    "longest_strike_below_mean": None,
    "count_above_mean": None,
    "count_below_mean": None,
    "first_location_of_maximum": None,
    "first_location_of_minimum": None,

    "sample_entropy": None,
    "binned_entropy": [{"max_bins": 10}],

    "linear_trend": [
        {"attr": "slope"},
        {"attr": "intercept"}
    ],

    "autocorrelation": [
        {"lag": 1},
        {"lag": 2},
    ],

    "fft_coefficient": [
        {"coeff": 0, "attr": "abs"},
        {"coeff": 1, "attr": "abs"},
    ],
}

def run(filename, chunk_size=None):
    data = pd.read_csv(filename, header=None, delimiter="\t")
    
    # convert to long format
    data_long = data.T.stack().reset_index()
    data_long = data_long.rename(columns={"level_0": "time", "level_1": "id", 0:"value"})
    
#    fc_parameters = MinimalFCParameters()
    
    # If chunk_size is None, process all at once
    if chunk_size is None:
        features = extract_features(
            data_long, 
            column_id="id", 
            column_sort="time", 
            default_fc_parameters = fc_parameters
        )
    else:
        features_list = []
        for start in range(0, data.shape[0], chunk_size):
            end = start + chunk_size
            chunk = data.iloc[start:end]
            #chunk_long = chunk.T.stack().reset_index()
            #print(chunk_long)            
            #chunk_long = chunk_long.rename(columns={"level_0": "time", "level_1": "id", 0:"value"})
            chunk_long = chunk.reset_index().melt(
                id_vars="index",
                var_name="time",
                value_name="value"
                ).rename(columns={"index": "id"})
            chunk_long["id"] = chunk_long["id"].astype(int)
            chunk_long["time"] = chunk_long["time"].astype(int)
            chunk_long["value"] = chunk_long["value"].astype(float)
            print(chunk_long.dtypes)
 
            chunk_features = extract_features(
                chunk_long, 
                column_id="id", 
                column_sort="time", 
                default_fc_parameters = fc_parameters,
            )
            features_list.append(chunk_features)
        features = pd.concat(features_list)
    
    impute(features)
    return features


def main():
    header_labels = ["target_cooler_condition_numeric", "target_valve_condition_numeric", "target_internal_pump_leakage_numeric", "target_hydraulic_bar_numeric", "target_stable_flag_binary"]

    labels = pd.read_csv("data/profile.txt", header = None, delimiter="\t")
    labels.columns = header_labels

    sensor_data_filepaths = [
                            "PS1.txt",
                            "PS2.txt",
                            "PS3.txt",
                            "PS4.txt",
                            "PS5.txt",
                            "PS6.txt",
                            "EPS1.txt",
                            "FS1.txt",
                            "FS2.txt",
                            "TS1.txt",
                            "TS2.txt",
                            "TS3.txt",
                            "TS4.txt",
                            "VS1.txt",
                            "CE.txt",
                            "CP.txt",
                            "SE.txt"]

    #chunk_size = 5
    for file in sensor_data_filepaths:
        features = run("data/" + file)
        dataset_final = pd.concat([features, labels] , axis = 1)
        dataset_final.to_csv(file.replace(".txt",".csv"), index = False)
        #print(dataset_final)
        #break


if __name__ == "__main__":
    main()

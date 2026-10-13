import argparse
import os
import tempfile
from pathlib import Path
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


SENSORS = ('PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2',
           'TS1', 'TS2', 'TS3', 'TS4', 'VS1', 'CE', 'CP', 'SE')
TARGET_COLUMNS = ('target_cooler_condition_regression', 'target_valve_condition_regression',
                  'target_internal_pump_leakage_regression', 'target_hydraulic_bar_regression',
                  'target_stable_flag_binary')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Extract the registered Hydraulic tsfresh features.')
    parser.add_argument('--sensor', choices=SENSORS)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    selected = (args.sensor,) if args.sensor else SENSORS
    outputs = {sensor: args.project_root / 'datasets' / ('EPS1' if sensor == 'CP' else sensor) / f'{sensor}.csv'
               for sensor in selected}
    for output in outputs.values():
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Refusing existing Hydraulic output: {output}')
        if any(parent.is_symlink() for parent in output.parents):
            raise ValueError('Hydraulic output must not use symlinked paths')
    labels = pd.read_csv(args.data_dir / 'profile.txt', header=None, delimiter='\t')
    labels.columns = list(TARGET_COLUMNS)
    for sensor, output in outputs.items():
        print(f'Hydraulic {sensor}: extracting tsfresh features', flush=True)
        features = run(args.data_dir / f'{sensor}.txt')
        frame = pd.concat([features, labels], axis=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f'.{sensor}-', suffix='.csv', delete=False) as handle:
                temporary = Path(handle.name)
            frame.to_csv(temporary, index=False)
            os.link(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        print(f'Hydraulic {sensor}: wrote {output}', flush=True)
    return 0

        #print(dataset_final)
        #break


if __name__ == "__main__":
    main()

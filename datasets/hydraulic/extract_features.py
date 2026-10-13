import os
import pandas as pd


def main():
    input_folder = "input_files"  # folder with your feature files
    labels_file = "data/profile.txt"  # labels file
    output_file = "features_with_labels.csv"

    # Column names for your label file
    header_labels = [
        "cooler_condition_numeric",
        "valve_condition_numeric",
        "internal_pump_leakage_numeric",
        "hydraulic_bar_numeric",
        "stable_flag_binary"
    ]

    # Load labels
    labels_df = pd.read_csv(labels_file, header=None, delimiter="\t")
    labels_df.columns = header_labels

    # List to store each feature file's dataframe
    all_dfs = []

    # Read and process each feature file
    for filename in sorted(os.listdir(input_folder)):
        if not filename.lower().endswith((".txt", ".dat")):
            continue
    
        file_path = os.path.join(input_folder, filename)
    
        # Read feature file (space-separated, no header)
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
    
        # Optional: rename columns to include filename prefix
        df.columns = [f"{filename}_f{i}" for i in range(df.shape[1])]
    
        all_dfs.append(df)

    # Concatenate all feature files horizontally
    features_df = pd.concat(all_dfs, axis=1)

    # Append the labels
    final_df = pd.concat([features_df, labels_df], axis=1)

    # Save the final CSV
    final_df.to_csv(output_file, index=False)
    print(f"Combined feature file with labels created: {output_file}")


if __name__ == "__main__":
    main()

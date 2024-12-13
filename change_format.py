import os
import glob
import numpy as np
import pandas as pd

# dir and file
data_path = "./result/"
csv_files = glob.glob(os.path.join(data_path, "nc_*.csv"))

# loop through each CSV file and process it
for file in csv_files:
    # obtain size
    size = int(file.split("_")[2].split("-")[0])
    # fixed params
    rng = np.random.RandomState(17)
    Q = 0.01 * np.diag(rng.random(size=size))
    p = 0.1 * rng.random(size=size)
    A = rng.normal(scale=0.1, size=(size, size))
    # read the CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    # check if column exists
    if "Constraints Viol" in df.columns:
        # convert list
        df["Param"] = df["Param"].apply(lambda x:eval(x))
        df["Sol"] = df["Sol"].apply(lambda x: eval(x) if pd.notna(x) else None)
        # compute violations
        df["Violations"] = df.apply(lambda row: np.maximum(0, A @ np.array(row["Sol"]) - np.array(row["Param"]))
                                if row["Sol"] is not None else None, axis=1)
        # calculate violation metrics
        df["Mean Violation"] = df["Violations"].apply(lambda viol: np.mean(viol) if viol is not None else None)
        df["Max Violation"] = df["Violations"].apply(lambda viol: np.max(viol) if viol is not None else None)
        df["Num Violations"] = df["Violations"].apply(lambda viol: np.sum(viol > 1e-6) if viol is not None else None)
        # drop unnecessary columns
        df = df.drop(columns=["Constraints Viol", "Violations"])
        df.to_csv(file)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SHUFFLE = False

def read_data() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    training_csv = pd.read_csv("../eleven_hour_data.csv")
    trimmed_training_csv = training_csv.loc[training_csv["vehicle_id"] == 5817.0][0:1000]
    # print(trimmed_training_csv)
    print("TRAINING DATA SIZE: ", len(trimmed_training_csv))

    trimmed_training_csv = trimmed_training_csv.dropna(axis=0)  # for no actual arrival time
    trimmed_training_csv = trimmed_training_csv.drop(["vehicle_id", "is_weekend"], axis=1)

    X = trimmed_training_csv.drop(["actual_arrival_time"], axis=1)
    y = trimmed_training_csv["actual_arrival_time"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.25, shuffle=SHUFFLE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, shuffle=SHUFFLE)

    return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == '__main__':
    U_train, Y_train, U_val, Y_val, U_test, Y_test = read_data()
    print(">", U_train.to_numpy())
    print(">", Y_train.to_numpy())
    print(">", U_val.to_numpy())
    print(">", Y_val.to_numpy())
    print(">", np.max(U_train.to_numpy(), axis=0))
    print(">", len(np.max(U_train.to_numpy(), axis=0)))




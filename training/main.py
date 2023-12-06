import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

ROW_DATA_SIZE = 50

training_csv = pd.read_csv("eleven_hour_data.csv")
training_csv["delta_lat"] = training_csv["stop_lat"].to_numpy()-training_csv["latitude"]
training_csv["delta_lon"] = training_csv["stop_lon"].to_numpy()-training_csv["longitude"]
training_csv.drop(["stop_lat", "stop_lon", "latitude", "longitude"], axis=1, inplace=True)
print(len(training_csv))

#[0:1000]  #
data = training_csv.loc[training_csv["vehicle_id"] == 5817.0][0:1000]

# rows = len(data)//ROW_DATA_SIZE + 1
# fig, axs = plt.subplots(nrows=rows)

# for i in range(0, rows):
#     data_inp = data[i*ROW_DATA_SIZE:i*ROW_DATA_SIZE+ROW_DATA_SIZE]
#
sns.pairplot(data) #, hue="actual_arrival_time")
plt.show()

# while (inp := input(">> ").upper()) != "QUIT":
#     data_inp = data[int(inp):int(inp)+50]
#     X = data_inp["current_time"]
#     Y = data_inp["actual_arrival_time"]
#
#     print(len(data_inp))
#
#     sns.pairplot(data_inp, y_vars=["actual_arrival_time"])
#     # plt.plot(X, Y, "o")
#
#     # plt.xlabel("Current Time")
#     # plt.ylabel("Actual Arrival Time")
#     plt.show()

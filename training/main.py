import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

training_csv = pd.read_csv("training4.csv")

data = training_csv.loc[training_csv["vehicle_id"] == 5881.0]
X = data["current_time"]
Y = data["actual_arrival_time"]

print(data)

# sns.pairplot(data)
plt.plot(X, Y, "o")

plt.show()

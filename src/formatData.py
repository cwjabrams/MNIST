import pandas as pd
import numpy as np

data = pd.read_csv("../data/train.csv")
data = data.values
np.save("../bin/training_data", data)

data = pd.read_csv("../data/test.csv")
data = data.values
np.save("../bin/test_data", data)

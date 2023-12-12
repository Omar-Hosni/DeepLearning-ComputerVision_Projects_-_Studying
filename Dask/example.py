import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")




arr = [100,200,300,400,500,600,700]

v = df[len(df)-50:]

v.to_csv("Virginica.csv", index=False)

vi = pd.read_csv("Virginica.csv")

vi.drop("Species", axis=1, inplace=True)
print(vi)
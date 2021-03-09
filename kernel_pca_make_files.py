import numpy as np
import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

np.random.seed(2)

# factor表示里圈和外圈的距离之比.每圈共有n_samples/2个点
x1, y1 = make_circles(n_samples=500, factor=0.2, noise=0.1)
ellipse = pd.DataFrame(x1)
ellipse.columns = ["x1", "y1"]
ellipse["category"] = y1
# save data in csv file
ellipse.to_csv("ellipse_dataset1.csv")

a = ellipse[ellipse["category"] == 0]
b = ellipse[ellipse["category"] == 1]
sea.scatterplot(data=a, x="x1", y="y1", color="r", label="a points", marker="+")
sea.scatterplot(data=b, x="x1", y="y1", color="b", label="b points", marker="x")
plt.show()

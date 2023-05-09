import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.stats import pearsonr

n = 100

x = 7.5 + 0.6 * np.random.normal(size=n)
y = -0.2*(x-7.5)**2 + 1 + 0.06 * np.random.normal(size=n)

data = {
    "x": x,
    "y": y
}

data = pd.DataFrame(data, columns=["cas odjezdu [h]","delka cesty [h]"])

print(f"pearsonov korelacny koeficient a p-hodnota{pearsonr(x,y)}")

plt.figure()
plt.plot(x, y,".")
plt.show()

a = 0
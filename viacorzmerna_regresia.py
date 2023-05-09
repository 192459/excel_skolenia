import numpy as np 
import pandas as pd 
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
import matplotlib.pyplot as plt 
import locale
locale.setlocale(locale.LC_NUMERIC, "de_DE")

def construct_X(x):
    output = np.column_stack((x))
    if output.shape[0]==1:
        output = output.T
    output = sm.add_constant(output)
    return output


beta0 = 25
beta1 = 8
sigma =3

np.random.seed(11111)

a = 1
b = 2


n = 100
x = np.random.uniform(low=-1,high=1, size=n)
y = np.random.uniform(low=-1,high=1, size=n)

y2 = x + 0.3*np.random.normal(size=n)

# X = construct_X(x)

# res = sm.OLS(y, X).fit()

# print(res.summary2())




# # plots 
# xx = np.linspace(a-0.4, b+0.4, 100)
# XX = construct_X(xx)


# prediction_full = res.get_prediction(XX).summary_frame(alpha=0.05)

# latex style 
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif", 
  "font.size":12
})

cm = 1/2.54  # centimeters in inches
plt.rcParams['axes.formatter.use_locale'] = True

print(np.corrcoef(x,y)[0,1])
print(np.corrcoef(x,y2)[0,1])

plt.figure()
plt.plot(x, y, ".")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()

plt.figure()
plt.plot(x, y2, ".")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()





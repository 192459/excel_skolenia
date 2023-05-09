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



x = np.array([1, 1.2, 1.7, 1.8, 1.9, 1.95, 1.97,  2.0])

n = len(x)

y = (x-(b-a))**2 + 0.05*np.random.normal(size=n)

X = construct_X(x)

res = sm.OLS(y, X).fit()

print(res.summary2())




# plots 
xx = np.linspace(a-0.4, b+0.4, 100)
XX = construct_X(xx)


prediction_full = res.get_prediction(XX).summary_frame(alpha=0.05)

# latex style 
plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif", 
  "font.size":12
})

cm = 1/2.54  # centimeters in inches
plt.rcParams['axes.formatter.use_locale'] = True

plt.figure()
plt.title("$y = \\beta_0 + \\beta_1 x $")
plt.plot(xx, prediction_full["mean"].values, label = "predikce")
plt.plot(x, y, ".", label = "pozorovani")
plt.fill_between(xx, prediction_full["obs_ci_lower"].values, prediction_full["obs_ci_upper"].values,  color = 'red', alpha = 0.1, label = 'konf. interval - hodnoty')
plt.fill_between(xx, prediction_full["mean_ci_lower"].values, prediction_full["mean_ci_upper"].values,  color = 'blue', alpha = 0.1, label = 'konf. interval - stredne hodnoty')
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.legend()
plt.show()





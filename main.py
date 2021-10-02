import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pt

P=np.array([1,2,3]).reshape((-1,1))
T=np.array([3.0,5.1,4.8])
model = LinearRegression().fit(P, T)
y = model.intercept_ + model.coef_ * P
print('Прогнозируемый ответ:', y)
fig, ax = pt.subplots()
pt.plot(P,y)
ax.scatter(P,y)
ax.scatter(P,T)
ax.grid()
pt.show()
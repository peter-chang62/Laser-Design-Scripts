from math import e
from scipy import integrate as inte
import numpy as np
import matplotlib.pyplot as plt


def f(t, x):
    return -x


solution = inte.RK45(f, 0, [1], np.inf)

# collect data
t_values = []
y_values = []
for i in range(100):
    # get solution step state
    solution.step()
    t_values.append(solution.t)
    y_values.append(solution.y[0])
    # break loop after modeling is finished
    if solution.status == "finished":
        break

data = zip(t_values, y_values)

import numpy as np
import matplotlib.pyplot as plt

def himmelblau(x, y):
    first_term = (x**2) + y - 11
    second_term = x + (y**2) - 7
    return (first_term**2) + (second_term) ** 2


x0 = 2
y0 = 1

k = 0.1
T0 = 1000
M = 300
N = 15
alpha = 0.85

f0 = himmelblau(x0, y0)

print("Initial X is %.3f" % x0)
print("Initial Y is %.3f" % y0)
print("Initial Z is %.3f" % f0)

temp = []
f_min = []

for i in range(M):
    for j in range(N):
        ran_x_1 = np.random.rand()
        ran_x_2 = np.random.rand()
        ran_y_1 = np.random.rand()
        ran_y_2 = np.random.rand()
        
        x1 = (1 - (2 * (ran_x_1 < 0.5))) * k * ran_x_2
        y1 = (1 - (2 * (ran_y_1 < 0.5))) * k * ran_y_2
        
        xt = x0 + x1
        yt = y0 + y1
        
        f_new = himmelblau(xt, yt)
        f_curr = himmelblau(x0, y0)
        
        if f_new <= f_curr:
            x0 = xt
            y0 = yt
        else:
            ran_1 = np.random.rand()
            formula = 1 / (np.exp((f_new - f_curr)/T0))
            if ran_1 <= formula:
                x0 = xt
                y0 = yt
    temp.append(T0)
    f_min.append()
    T0 *= alpha
    
print("X is %.3f" % x0)
print("Y is %.3f" % y0)
print("Final Objective Function value is %.3f" % f_curr)

plt.plot(temp, f_min)
plt.title("Himmelblau vs. Temperature", fontsize=20, fontweight="bold")
plt.xlabel("Temperature", fontsize=18, fontweight="bold")
plt.ylabel("Himmelblau", fontsize=18, fontweight="bold")

plt.xlim(1000, 0)
plt.xticks(np.arrange(min(temp), max(temp), 100), fontweight="bold")
plt.yticks(fontweight="bold")

plt.show()

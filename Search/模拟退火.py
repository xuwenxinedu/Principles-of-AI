import numpy as np
import matplotlib.pyplot as plt
X_BOUND = [-1, 2]

def F(x):
	return x * np.sin(10 * np.pi * x) + 1.0
def fire():
	x=np.random.uniform(-1,2,100)
	ans = -10000000
	t=3
	points = []
	while t>1e-5:
		f_old = F(x)
		next_x = x + (np.random.rand(1)*2-1)*0.1*t
		f_next = F(next_x)
		for i in range(len(x)):
			if -1<=next_x[i]<=2:
				if f_old[i]<f_next[i]:
					x[i]=next_x[i]
				else:
					p=np.e**((f_next[i]-f_old[i])/t)
					if np.random.rand()<p:
						x[i]=next_x[i]
		t = t*0.98
		points.append(x[np.argmax(f_old)])
	ans = f_old[np.argmax(f_old)] if f_old[np.argmax(f_old)]>ans else ans
	return points
plt.ion()
x = np.linspace(*X_BOUND,2000)
plt.plot(x, F(x))
point=fire()
point=np.array(point)
sca = plt.scatter(point,F(point))
print(F(point[np.argmax(F(point))]))
plt.ioff()
plt.show()

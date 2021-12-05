import numpy as np
import matplotlib.pyplot as plt
#变量声明
DNA_SIZE = 10
POP_SIZE = 200
CROSS_RATE = 0.95
N_GENERATION = 3000
X_BOUND = [-1, 2]
MUTATE_RATE = 0.005


#计算函数
def f(x):
    return x * np.sin(10 * np.pi * x) + 1.0

def dna2num(pop):
    return (pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*X_BOUND[1])

def bianyi(pop):
    for elem in pop:
        for i in range(DNA_SIZE):
            if np.random.rand()<MUTATE_RATE:
                elem[i]=0 if elem[i]==1 else 1
    return pop

def jiaopei(pop):
    pop = pop.tolist()
    poplen = len(pop)
    for i in range(0,poplen-1,2):
        point = np.random.randint(1,9)
        temp=pop[i]
        temp1=pop[i+1]
        if np.random.rand()<CROSS_RATE:
            pop[i][point:10]=temp1[point:10]
            pop[i+1][0:point]=temp[0:point]
    pop = np.array(pop)
    return pop

def select(pop, fitness):
    index = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p = fitness/fitness.sum())
    return pop[index]

def getfit(pred):
    return pred-np.min(pred)+1e-3

pop = np.random.randint(0,2,(1,2000)).reshape(200,10)
dianji=[]
x = np.linspace(*[-1,2],2000)
plt.plot(x,f(x))#绘图

for aaaaa in range(N_GENERATION):
    fval = f(dna2num(pop))
    fit = getfit(fval)
    print('this time the best DNA is:',pop[np.argmax(fit)])
    dianji.append(pop[np.argmax(fit)])
    print('')
    pop = select(pop,fit)
    pop = jiaopei(pop)
    pop = bianyi(pop)
dianji = np.array(dianji)
fval = f(dna2num(dianji))
sca = plt.scatter(dna2num(dianji),fval)
fit = getfit(fval)
dix = dianji[np.argmax(fit)]
dix = dna2num(dix)
print(str(dix)+','+str(f(dix)))

plt.show()
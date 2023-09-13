import matplotlib.pyplot as plt
import numpy as np

epochs = [i+1 for i in range(15)]

entityf1 = []
relationf1 = []

entityf1_AT = []
relationf1_AT = []

with open("end.log") as f:
    for i, data  in enumerate(f):
        if i % 2 == 0:
            if i == 15 * 2:
                break
            entityf1.append(float(data[-7:-1])*100)
        else:
            relationf1.append(float(data[-7:-1])*100)
            if i == 15 * 2:
                break
with open("end.txt") as f:
    for i, data in enumerate(f):
        if i % 2 == 0:
            if i == 15 * 2:
                break
            entityf1_AT.append(float(data[-6:-1])*100)
        else:
            relationf1_AT.append(float(data[-6:-1])*100)
            if i == 15 * 2:
                break

print(entityf1)
print(relationf1)
# print(len(entityf1))
print(len(relationf1))
print(epochs)

print(entityf1_AT)
print(relationf1_AT)
print(len(entityf1))
print(len(relationf1_AT))



plt.xlabel("epochs")
plt.ylabel("%")


plt.plot(epochs, entityf1)
plt.plot(epochs, relationf1)
plt.plot(epochs, entityf1_AT)
plt.plot(epochs, relationf1_AT)
plt.show()
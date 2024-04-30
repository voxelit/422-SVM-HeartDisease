import matplotlib.pyplot as plt

Folds = [1,2,3,4,5]
Accuracy = [0.876070, 0.876700, 0.881554, 0.876128, 0.887062]
WeightedAverage = [0.9, 0.9, 0.9, 0.9, 0.91]

plt.plot(Folds, Accuracy, label='Accuracy')
plt.plot(Folds, WeightedAverage, label='Weighted Average')
plt.title('SVM Accuracy')
plt.xlabel('Folds')
plt.ylabel('Percentage')
plt.legend()
plt.show()
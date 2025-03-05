# %% [markdown]
# # 1.1 Data Preparation

# %%
import numpy as np
from pandas import read_csv

# %%
def parse_csv(filepath: str):
    data = read_csv(filepath)

    # targets = array of target values (last column)
    targets = data.values[:, -1].astype(float)

    # inputs = array of input values (all columns except last)
    inputs = data.values[:, :-1].astype(float)

    return np.array(inputs), np.array(targets)

# %%
inputs, targets = parse_csv('glass-dataset.csv')

# %%
# This is just to prove its working
print(inputs[:5]);

# %% [markdown]
# # 1.2 Classification

# %%
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# %%
def pca(inputs):
    pca = PCA(n_components=2)

    return pca.fit_transform(inputs)

# %%
compressed = pca(inputs)

# %%
def knn(inputs, targets, k: int):
    classifier = KNeighborsClassifier(n_neighbors=k)

    classifier.fit(inputs, targets)

    return classifier.predict(inputs)

# %%
def how_good_is_k(inputs, targets, k: int):
    classifiedData = knn(inputs, targets, k)

    totalRight = 0
    totalEntries = len(targets)

    for i in range(totalEntries):
        if classifiedData[i] == targets[i]:
            totalRight += 1

    return totalRight / totalEntries

# %%
# Returns the most accurate k values' classified data
def classify_knn(inputs, targets):
    accuracies = []

    for k in range(1, 20):
        accuracies.append(how_good_is_k(inputs, targets, k))

    plt.figure()
    plt.plot(range(1, 20), accuracies)
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.title("Accuracy of kNN with different k values")
    plt.show()

    return knn(inputs, targets, accuracies.index(max(accuracies)) + 1)

# %%
classifiedData = classify_knn(compressed, targets)

# %%
plt.figure()
plt.scatter(compressed[:, 0], compressed[:, 1], c=classifiedData)
plt.show()

# %%
# kernel can be "linear", "poly" or "rbf"
# degree can be 2 or 3
def svm(inputs, targets, kernel: str, degree: int):
    classifier = SVC(kernel=kernel, degree=degree)

    classifier.fit(inputs, targets)

    return classifier.predict(inputs)

# %%
def how_good_is_svm(inputs, targets, kernel, degree):
    classifiedData = svm(inputs, targets, kernel, degree)

    totalRight = 0
    totalEntries = len(targets)

    for i in range(totalEntries):
        if classifiedData[i] == targets[i]:
            totalRight += 1

    return totalRight / totalEntries

# %%
def classify_svm(inputs, targets):
    kernels = ["linear", "poly", "rbf"]
    degrees = [2, 3]
    accuracies = []

    for kernel in kernels:
        for degree in degrees:
            accuracies.append([kernel, degree, how_good_is_svm(inputs, targets, kernel, degree)])

    # this takes the 3rd item from the each item in the list and uses it as the key to get the highest acc.
    mostAccurate = max(accuracies, key=lambda x: x[2])

    return svm(inputs, targets, mostAccurate[0], mostAccurate[1])

# %%
classifiedData = classify_svm(inputs, targets)

# %%
plt.figure()
plt.scatter(compressed[:, 0], compressed[:, 1], c=classifiedData)
plt.show()

# %% [markdown]
#

# %% [markdown]
# # 1.3 Assessment of Classification

# Question:
#
# After identifying the best model parameters in the previous task, the
# classification models you have implemented must be assessed. To do this you
# are required to assess the accuracy for each model. You may use the accuracy
# implementation available to do this. It is not sufficient to report a single
# accuracy score. You must use cross-validation to report training results and
# report these values using a plot. You will also need to write a summary
# analysing your results and findings.

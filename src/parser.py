import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Parser:

    def normalize_all(self, X):
        # normalize the database
        # by mean and standard desviation

        result = []

        for i in range(len(X)):
            result.append(self.normalize(X[i]))

        return np.asarray(result)

    def normalize(self, x):
        # normalize the sample
        # by mean and standard desviation

        result = []

        for i in range(int(len(x) - 1)):
            if ((i % 2) == 0):
                result.append([x[i], x[i + 1]])

        result = np.asarray(result)

        mean_x = np.mean(result[:, 0])
        mean_y = np.mean(result[:, 1])
        mean = np.array([mean_x, mean_y])

        std_x = np.std(result[:, 0])
        std_y = np.std(result[:, 1])
        std = np.array([std_x, std_y])

        result = result - mean
        result = result / std

        #print("Media: " + str(mean))
        #print("std: " + str(std))

        return np.reshape(result, [len(x)])

    def pca_all(self, X):
        # project pca on the database

        result = []

        for i in range(len(X)):
            result.append(self.pca(X[i]))

        return np.asarray(result)

    def pca(self, x):
        # project pca for a sample

        result = []

        for i in range(int(len(x) - 1)):
            if ((i % 2) == 0):
                result.append([x[i], x[i + 1]])

        result = np.asarray(result)

        #print(result)

        pca_model = PCA(n_components=2)
        result = pca_model.fit_transform(result)

        return np.reshape(result, [len(x)])


    def grouping_all(self, X, num_groups):
        # grouping samples together

        result = []

        for i in range(len(X)):
            result.append(self.grouping(X[i], num_groups))

        return np.asarray(result)


    def grouping(self, x, num_groups):
        # grouping samples together
        result = []

        for i in range(int(len(x) - 1)):
            if ((i % 2) == 0):
                result.append([x[i], x[i + 1]])

        result = np.asarray(result)


        # to take each group representing
        tam_grupo = int(len(result) / num_groups)
        result2 = []
        for i in range(len(result)):

            if ((i%tam_grupo) == 0):
                x = np.mean(result[i:i+tam_grupo, 0])
                y = np.mean(result[i:i + tam_grupo, 1])
                result2.append([x, y])

        result2 = np.asarray(result2)

        result2 = np.reshape(result2, [num_groups*2])
        return result2

    def plot(self, x):
        # plot sample

        plt.close()

        for i in range(int(len(x) - 1)):
            if ((i % 2) == 0):
                plt.scatter(x[i], x[i+1])
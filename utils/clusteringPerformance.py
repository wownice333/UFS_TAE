from sklearn.cluster import KMeans
from scipy.stats import mode
import warnings
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")


### K-means clustering
def Clustering(features, gnd, clusterNum, randNum, typ='KMeans'):
    if typ == 'KMeans':
        kmeans = KMeans(n_clusters=clusterNum, n_init=1, max_iter=500,
                        random_state=randNum)
        estimator = kmeans.fit(features)
        clusters = estimator.labels_
        label_pred = estimator.labels_
        labels = np.zeros_like(clusters)
        for i in range(clusterNum):
            mask = (clusters == i)
            labels[mask] = mode(gnd[mask])[0]

        return labels, label_pred


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    np.asarray(ind)
    ind = np.transpose(ind)
    # print(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size



### Evaluation metrics of clustering performance
def clusteringMetrics(trueLabel, predictiveLabel):
    # Clustering accuracy
    ACC = cluster_acc(trueLabel, predictiveLabel)

    # Normalized mutual information
    NMI = metrics.v_measure_score(trueLabel, predictiveLabel)

    # Adjusted rand index
    ARI = metrics.adjusted_rand_score(trueLabel, predictiveLabel)

    return ACC, NMI, ARI


### Report mean and std of 10 experiments
def StatisticClustering(features, gnd, typ='KMeans'):
    ### Input the mean and standard diviation with 10 experiments
    repNum = 10
    ACCList = np.zeros((repNum, 1))
    NMIList = np.zeros((repNum, 1))
    ARIList = np.zeros((repNum, 1))

    clusterNum = int(np.max(gnd)) - int(np.min(gnd)) + 1

    for i in range(repNum):

        predictiveLabel, label_pred = Clustering(features, gnd, clusterNum, i, typ)

        ACC, NMI, ARI = clusteringMetrics(gnd, predictiveLabel)
        ACCList[i] = ACC
        NMIList[i] = NMI
        ARIList[i] = ARI

    ACCmean_std = np.around([np.mean(ACCList), np.std(ACCList)], decimals=4)
    NMImean_std = np.around([np.mean(NMIList), np.std(NMIList)], decimals=4)
    ARImean_std = np.around([np.mean(ARIList), np.std(ARIList)], decimals=4)

    return ACCmean_std, NMImean_std, ARImean_std

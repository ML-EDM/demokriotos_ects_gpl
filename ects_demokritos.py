import numpy as np

from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

class ECTS:

    """
    Early classification on time series(2012)
    Code has been taken from: https://github.com/Eukla/ETS/blob/master/ets/algorithms/ects.py
    and made compatible with the ml-edm library: https://github.com/ML-EDM/ml_edm
    """

    def __init__(self, 
                 timestamps, 
                 support=0, 
                 relaxed=False,
                 n_jobs=1):
        """
        Creates an ECTS instance.

        :param timestamps: a list of timestamps for early predictions
        :param support: minimum support threshold
        :param relaxed: whether we use the Relaxed version or the normal
        """
        
        super().__init__()
        
        ######Constant attributes#######
        self.require_past_probas = False
        self.require_classifiers = False
        self.alter_classifiers = False
        ################################

        self.rnn = dict()
        self.nn = dict()
        self.mpl = dict()
        self.timestamps = timestamps
        self.support = support
        self.clusters = dict()
        self.occur = dict()
        self.relaxed = relaxed
        self.correct = None

        self.n_jobs = n_jobs

    def fit(self, X, y):

        """
        Function that trains the model using Agglomerating Hierarchical clustering

        :param train_data: a Dataframe containing-series
        :param labels: a Sequence containing the labels of the data
        """
        self.data = X

        self.labels = y
        if self.relaxed:
            self.__leave_one_out()

        indexes, values = np.unique(self.labels, return_counts=True)
        for i, index in enumerate(indexes):
            self.occur[index] = values[i]

        # Finding the RNN of each item
        time_pos = 0
        for e in self.timestamps:
            product = self.__nn_non_cluster(e)  # Changed to timestamps position
            self.rnn[e] = product[1]
            self.nn[e] = product[0]
            time_pos += 1

        temp = {}
        finished = {}  # Dictionaries that signifies if an mpl has been found
        for e in reversed(self.timestamps):
            for index, _ in enumerate(self.data):

                if index not in temp:
                    self.mpl[index] = e
                    finished[index] = 0  # Still MPL is not found
                else:
                    if finished[index] == 1:  # MPL has been calculated for this time-series so nothing to do here
                        continue

                    if self.rnn[e][index] is not None:
                        self.rnn[e][index].sort()
                    # Sorting it in order to establish that the RNN is in the same order as the value
                    if temp[index] is not None:
                        temp[index].sort()

                    if self.rnn[e][index] == temp[index]:  # Still going back the timestamps
                        self.mpl[index] = e
                    else:  # Found k-1
                        finished[index] = 1  # MPL has been found!

                temp[index] = self.rnn[e][index]

        self.__mpl_clustering()
        
        return self

    def __leave_one_out(self):
        nn = []
        for index, row in enumerate(self.data):  # Comparing each time-series

            data_copy = self.data.copy()
            data_copy = np.delete(data_copy, index, axis=0)

            for index2, row2 in enumerate(data_copy):

                temp_dist = np.linalg.norm(row - row2)

                if not nn:
                    nn = [(self.labels[index2], temp_dist)]
                elif temp_dist >= nn[0][1]:
                    nn = [(self.labels[index2], temp_dist)]

            if nn[0][0] == self.labels[index]:
                if not self.correct:
                    self.correct = [index]
                else:
                    self.correct.append(index)
            nn.clear()

    def __nn_non_cluster(self, prefix):

        """Finds the NN of each time_series and stores it in a dictionary

        :param prefix: the prefix with which we will conduct the NN

        :return: two dicts holding the NN and RNN"""

        nn = {}
        rnn = {}

        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(self.data[:, 0:prefix + 1])
        def something(row):
            return neigh.kneighbors([row])

        result_data = np.apply_along_axis(
            something, axis=1, arr=self.data[:, 0:prefix + 1]
        )
        for index, value in enumerate(result_data):
            value = (value[0], value[1])
            if index not in nn:
                nn[index] = []
            if index not in rnn:
                rnn[index] = []
            for item in value[1][0]:
                if item != index:
                    nn[index].append(item)
                    if item not in rnn:
                        rnn[item] = [index]
                    else:
                        rnn[item].append(index)
        
        return nn, rnn

    def __cluster_distance(self, cluster_a, cluster_b):

        """
        Computes the distance between two clusters as the minimum among all
        inter-cluster pair-wise distances.

        :param cluster_a: a cluster
        :param cluster_b: another cluster
        :return: the distance
        """

        min_distance = np.inf
        for i in cluster_a:
            for j in cluster_b:
                d = np.linalg.norm(self.data[i] - self.data[j])
                if min_distance > d:
                    min_distance = d

        return min_distance

    def nn_cluster(self, cl_key, cluster_index):

        """Finds the nearest neighbor to a cluster
        :param cluster_index: List of indexes contained in the list
        :param cl_key: The key of the list in the cluster dictionary
        """
        #global x
        dist = np.inf
        candidate = []  # List that stores multiple candidates

        for key, value in self.clusters.items():  # For each other cluster

            if cl_key == key:  # Making sure its a different to our current cluster
                continue
            temp = self.__cluster_distance(cluster_index, value)  # Find their Distance

            if dist > temp:  # If its smaller than the previous, store it
                dist = temp
                candidate = [key]

            elif dist == temp:  # If its the same, store it as well
                candidate.append(key)
        #x-=1
        return candidate

    def __rnn_cluster(self, e, cluster):

        """
        Calculates the RNN of a cluster for a certain prefix.

        :param e: the prefix for which we want to find the RNN
        :param cluster: the cluster that we want to find the RNN
        """

        rnn = set()
        complete = set()
        for item in cluster:
            rnn.union(self.rnn[e][item])
        for item in rnn:
            if item not in cluster:
                complete.add(item)
        return complete

    def __mpl_calculation(self, cluster):

        """Finds the MPL of discriminative clusters
        
        :param cluster: The cluster of which we want to find it's MPL"""

        # Checking if the support condition is met
        index = self.labels[cluster[0]]
        if self.support > len(cluster) / self.occur[index]:
            return
        mpl_rnn = self.timestamps[len(self.timestamps) - 1]  
        # Initializing the  variables that will indicate the 
        # minimum timestamp from which each rule applies
        mpl_nn = self.timestamps[len(self.timestamps) - 1]
        """Checking the RNN rule for the clusters"""
        
        # Finding the RNN for the L
        curr_rnn = self.__rnn_cluster(self.timestamps[len(self.timestamps) - 1], cluster)

        if self.relaxed:
            curr_rnn = curr_rnn.intersection(self.correct)

        for e in reversed(self.timestamps):

            temp = self.__rnn_cluster(e, cluster)  # Finding the RNN for the next timestamp
            if self.relaxed:
                temp = temp.intersection(self.correct)
            # If their division is an empty set, then the RNN is the same so the
            if not curr_rnn - temp:  
                # MPL is e
                mpl_rnn = e
            else:
                break
            curr_rnn = temp

        """Then we check the 1-NN consistency"""
        rule_broken = 0
        for e in reversed(self.timestamps):  # For each timestamp

            for series in cluster:  # For each time-series

                for my_tuple in self.nn[e][series]:  # We check the corresponding NN to the series
                    if my_tuple not in cluster:
                        rule_broken = 1
                        break
                if rule_broken == 1:
                    break
            if rule_broken == 1:
                break
            else:
                mpl_nn = e
        for series in cluster:
            pos = max(mpl_rnn, mpl_nn)  # The value at which at least one rule is in effect
            if self.mpl[series] > pos:
                self.mpl[series] = pos

    def __mpl_clustering(self):

        """Executes the hierarchical clustering"""
        n = self.data.shape[0]
        redirect = {}  # References an old cluster pair candidate to its new place
        discriminative = 0  # Value that stores the number of discriminative values found
        """Initially make as many clusters as there are items"""
        for index in range(len(self.data)):
            self.clusters[index] = [index]
            redirect[index] = index

        result = []
        """Clustering loop"""
        while n > 1:  # For each item
            closest = {}
            my_list = list(self.clusters.items())
            res = Parallel(n_jobs=self.n_jobs) \
                (delayed(self.nn_cluster)(k, idx) for k, idx in my_list)
            
            for key,p  in zip(self.clusters.keys(),res):
                closest[key] = p

            for key, value in closest.items():
                for item in list(value):
                    if key in closest[item]:  # Mutual pair found
                        closest[item].remove(key)
                        #If 2 time-series are in the same cluster
                        # (in case they had an 3d  neighboor that invited them in the cluster)
                        if  redirect[item]==redirect[key]:  
                            continue
                        for time_series in self.clusters[redirect[item]]:
                            self.clusters[redirect[key]].append(time_series)  # Commence merging
                        del self.clusters[redirect[item]]
                        n = n - 1
                        redirect[item] = redirect[key]  # The item can now be found in another cluster
                        for element in self.clusters[redirect[key]]:  # Checking if cluster is discriminative
                            result.append(self.labels[element])

                        x = np.array(result)
                        if len(np.unique(x)) == 1:  # If the unique class labels is 1, then the
                            # cluster is discriminative
                            discriminative += 1
                            self.__mpl_calculation(self.clusters[redirect[key]])

                        for neighboors_neigboor in closest:  # The items in the cluster that has been assimilated can
                            # be found in the super-cluster
                            if redirect[neighboors_neigboor] == item:
                                redirect[neighboors_neigboor] = key
                        result.clear()
                        
            if discriminative == 0:  # No discriminative clusters found
                break
            discriminative = 0

    def predict(self, X):

        """
        Prediction phase.
        Finds the 1-NN of the test data and if the MPL oof the closest 
        time-series allows the prediction, then return that prediction
         """
        
        predictions, triggers, times_idx = [], [], []
        nn = []
        candidates = []  # will hold the potential predictions
        cand_min_mpl = []

        for test_row in X:
            for e in self.timestamps:
                neigh = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data[:, 0:e+1])
                neighbors = neigh.kneighbors([test_row[0:e+1]])
                candidates.clear()
                cand_min_mpl.clear()
                nn = neighbors[1]
                for i in nn:
                    if e >= self.mpl[i[0]]:
                        candidates.append((self.mpl[i[0]], self.labels[i[0]]))  # Storing candidates by mpl and by label
                if len(candidates) > 1:  # List is not empty so wee found candidates
                    candidates.sort(key=lambda x: x[0])
                    for candidate in candidates:
                        if candidate[0] == candidates[0][0]:
                            cand_min_mpl.append(candidate)  # Keeping the candidates with the minimum mpl
                        else:
                            break  # From here on the mpl is going to get bigger
                    predictions.append(max(set(cand_min_mpl), key=cand_min_mpl.count))  # The second argument is the max label
                    triggers.append(True)
                    times_idx.append(e)
                    break
                elif len(candidates) == 1:  # We don't need to to do the above if we have only one nn
                    predictions.append(candidates[0][1])
                    triggers.append(True)
                    times_idx.append(e)
                    break

            if len(candidates) == 0:
                triggers.append(False)
                predictions.append(np.nan)
                times_idx.append(self.timestamps[-1])

        return np.array(predictions), np.array(triggers), np.array(times_idx)
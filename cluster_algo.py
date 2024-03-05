from collections import Counter
from sklearn.metrics import (
    davies_bouldin_score,
    silhouette_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ClusteringAlgorithm:
    def __init__(self, data, threshold_value):
        self.data_f = data
        self.correlation_matrix = data.corr(method='spearman')
        self.threshold_value = threshold_value
        self.labels = None
        self.error = None
        self.centroid = None
        self.no_of_clusters = None
        self.data_length = self.correlation_matrix.shape[0]
        self.make_clusters()

    def make_clusters(self):
        self.labels = self.optimize_clusters(self.threshold_value)

    def get_centroid(self):
        self.centroid = self.find_centroid(self.labels)
        return self.centroid

    def get_error(self):
        if self.centroid is None:
            self.get_centroid()
        self.error = self.calc_error(self.labels, self.centroid)
        return self.error

    def find_overlayers(self, labels):
        d = {}
        for i in labels:
            d.setdefault(i, 0)
            d[i] += 1
        return [k for k, v in d.items() if v < 2]

    def calc_error(self, labels, g_centre):
        data_f = self.data_f
        data_f["label"] = labels
        overlayers = self.find_overlayers(labels)
        data_f = data_f[~data_f["label"].isin(overlayers)]
        data_g = data_f.groupby("label", sort=True)
        m = len(data_f.columns)
        n = len(data_g)

        error = {}

        for idx, df in data_g:
            error[idx] = self.rms(df.drop("label", axis=1), g_centre.iloc[idx])

        return round(sum(error[idx] for idx in error) / n / m, 6)

    def rms(self, ser, g_centre):
        diff = ser.subtract(g_centre)
        return np.sqrt((diff ** 2).mean())

    def find_centroid(self, labels):
        data_f = self.data_f
        data_f["label"] = labels
        data_g = data_f.groupby("label")
        return data_g.mean()

    def optimize_clusters(self, threshold_value):
        common_fields = self.get_relatable_fields(
            self.correlation_matrix.values, self.correlation_matrix.columns, threshold_value)
        common_clusters = common_fields.copy()
        res2 = self.relatable_val(common_fields)

        while True:
            rel_pair, max1 = self.get_max_full(res2.values, threshold_value)
            if max1 == 0:
                break
            if rel_pair is not None:  # Check if rel_pair is not None
                key1, key2 = int(rel_pair[0] or 0), int(rel_pair[1] or 0)  # Convert keys to integers

                common_fields[key1] = list(set(common_fields[key1]).union(set(common_fields[key2])))
                

                res2 = pd.DataFrame(
                    self.relatable_val(common_fields), index=range(len(common_fields)))

        final_list = sorted(common_fields, key=lambda x: len(x), reverse=True)

        final_res = set()
        col = []
        total_length = len(self.correlation_matrix.columns)
        for k, v in enumerate(final_list):
            temp = final_res.union(v)
            if temp != final_res:
                final_res = temp
                col.append(k)
            if len(final_res) == total_length:
                final_list = [final_list[i] for i in col]
                break
        final_list = [item for item in final_list if len(item) > 1]

        labels = self.get_labels(final_list, common_clusters)
        return labels

    def get_labels(self, clusters, original_clusters):
        labels = [-1 for i in range(self.data_length)]
        c = 0
        for ln in clusters:
            for v in ln:
                if labels[v] == -1:
                    labels[v] = c
            c += 1

        label_count = Counter(labels)
        d = {}
        for lbl in range(self.data_length):
            if label_count[labels[lbl]] == 1:
                for k in original_clusters[lbl]:
                    if k == lbl:
                        continue
                    d.setdefault(labels[k], 0)
                    d[labels[k]] += 1

                s = sum(d.values())

                for k, i in d.items():
                    if round(i/s, 2) >= 0.75:
                        labels[lbl] = k
                        break
                else:
                    labels[lbl] = -1
                d.clear()

        label_count = Counter(labels)
        c = 0
        d = {}
        for k in label_count.keys():
            if k != -1:
                d[k] = c
                c += 1

        for i in range(self.data_length):
            if labels[i] == -1:
                labels[i] = c
                c += 1
            else:
                labels[i] = d[labels[i]]

        self.no_of_clusters = c
        return labels

    def get_max_full(self, matrix, threshold_value):
        np.fill_diagonal(matrix, 0)
        max_indices = np.unravel_index(
            np.argmax(matrix), matrix.shape)
        max_value = matrix[max_indices]

        if max_value < threshold_value:
            return (None, None), 0

        return max_indices, max_value

    def get_relatable_fields(self, matrix, header, threshold_value):
        greater_avg_dict = []
        temp = set()
        for v in matrix:
            avg = self.get_avg_row(v)
            for val, j in enumerate(v):
                if j >= avg and j >= threshold_value:
                    temp.add(header[val])
            greater_avg_dict.append(temp.copy())
            temp.clear()
        return greater_avg_dict

    def relatable_val(self, greater_avg_dict):
        rows = len(greater_avg_dict)
        tranform_dict = np.zeros((rows, rows))
        for k1, v1 in enumerate(greater_avg_dict):
            for k2, v2 in enumerate(greater_avg_dict):
                n1 = len(set(v1).union(set(v2)))
                n2 = len(set(v1).intersection(set(v2)))
                d = n2 / n1
                tranform_dict[k1, k2] = d

        return pd.DataFrame(tranform_dict, index=range(rows))

    def get_avg_row(self, matrix_row):
        return (sum(matrix_row) - 1) / (len(matrix_row) - 1)


def n_cluster_graph(n_clusters, threshold_values):
    plt.figure(figsize=(6, 5))

    # Plot Data
    plt.plot(threshold_values, n_clusters, marker='o', linestyle='--')
    plt.plot(threshold_values, [2]*len(threshold_values))
    plt.plot(threshold_values, [6]*len(threshold_values))

    # Formatting
    plt.grid()
    plt.xlabel("Threshold")
    plt.ylabel("No of clusters")


def calculate_rate_of_change(errors):
    return np.diff(errors)


# def error_graph(errors, threshold_values):
#     roc = calculate_rate_of_change(errors)
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('Threshold Values')
#     ax1.set_ylabel('Error', color=color)
#     ax1.plot(threshold_values, errors, color=color, marker='o')
#     ax1.tick_params(axis='y', labelcolor=color)

#     plt.grid()

#     ax2 = ax1.twinx()
#     color = 'tab:blue'
#     ax2.set_ylabel('Rate of Change', color=color)
#     ax2.plot(threshold_values[:-1], roc, color=color, marker='*')
#     ax2.tick_params(axis='y', labelcolor=color)
#     plt.grid()
#     fig.tight_layout()


def show_optimum_value(errors, threshold_values):
    x1 = 0.23
    x2 = 0.25
    plt.figure(figsize=(6, 5))

    plt.plot(threshold_values, errors, marker='o', label='Error')
    plt.axvline(x=x1, color='r', linestyle='--', label='Potential Elbow Point 1')
    plt.axvline(x=x2, color='g', linestyle='--', label='Potential Elbow Point 2')

    plt.xlabel('Threshold Values')
    plt.ylabel('Error')
    plt.legend()
    return x1, x2


def add_index(res, Y, name, labels, actual_labels):
    res[(name, max(labels)+1)] = {
        "DB": round(davies_bouldin_score(Y, labels), 4),
        "SI": round(silhouette_score(Y, labels), 4),
        "CH": round(calinski_harabasz_score(Y, labels), 4),
        "ARI": round(adjusted_rand_score(actual_labels, labels), 4),
        "NMI": round(normalized_mutual_info_score(actual_labels, labels), 4)
    }


# def compare_algos(res):
#     X = list(map(lambda x: x[0][:4], res.keys()))
#     db = [i['DB'] for i in res.values()]
#     si = [i['SI'] for i in res.values()]
#     ch = [i['CH'] for i in res.values()]
#     ari = [i['ARI'] for i in res.values()]
#     nmi = [i['NMI'] for i in res.values()]
#     X_axis = np.arange(len(X))
#     size = (5, 4)
#     width = 0.4

#     fig, axs = plt.subplots(2, 3, figsize=(size[0] * 3, size[1] * 2), gridspec_kw={
#                             'hspace': 0.5, 'wspace': 0.5})

#     axs[0, 0].bar(X_axis, db, width, label='DB Index', color='green')
#     axs[0, 0].set_xticks(X_axis)
#     axs[0, 0].set_xticklabels(X)
#     axs[0, 0].set_xlabel("Threshold value")
#     axs[0, 0].set_ylabel("DB Index")

#     axs[0, 1].bar(X_axis, si, width, label='Silhouette Score', color='blue')
#     axs[0, 1].set_xticks(X_axis)
#     axs[0, 1].set_xticklabels(X)
#     axs[0, 1].set_xlabel("Threshold value")
#     axs[0, 1].set_ylabel("Silhouette Score")

#     axs[0, 2].bar(X_axis, ch, width, label='Calinski Harabasz Score', color='cyan')
#     axs[0, 2].set_xticks(X_axis)
#     axs[0, 2].set_xticklabels(X)
#     axs[0, 2].set_xlabel("Threshold value")
#     axs[0, 2].set_ylabel("Calinski Harabasz Score")

#     axs[1, 0].bar(X_axis, ari, width, label='Adjusted Rand Score', color='yellow')
#     axs[1, 0].set_xticks(X_axis)
#     axs[1, 0].set_xticklabels(X)
#     axs[1, 0].set_xlabel("Threshold value")
#     axs[1, 0].set_ylabel("Adjusted Rand Score")

#     axs[1, 1].bar(X_axis, nmi, width, label='Normalized Rand Score', color='red')
#     axs[1, 1].set_xticks(X_axis)
#     axs[1, 1].set_xticklabels(X)
#     axs[1, 1].set_xlabel("Threshold value")
#     axs[1, 1].set_ylabel("Normalized Rand Score")

#     fig.delaxes(axs[1, 2])
#     plt.tight_layout()
#     plt.show()


def plot_graph(n_clusters, threshold_values, errors):
    n_cluster_graph(n_clusters, threshold_values)
    # error_graph(errors, threshold_values)
    vals = show_optimum_value(errors, threshold_values)
    plt.show()
    return vals

# Save the Clusters and their centroid vectors having headers Cluster,NoOfStudentsInCluster, v1,v2,..........v_n where n is the dimension of vector
def save_clusters_and_centroids(data_f, labels, centroid, file_name):
    data_f["label"] = labels
    data_g = data_f.groupby("label")
    with open(file_name, 'w') as f:
        f.write("Cluster,NoOfStudentsInCluster," + ",".join(data_f.columns) + "\n")
        for idx, df in data_g:
            f.write(f"{idx},{len(df)},")
            f.write(",".join([str(i) for i in centroid.iloc[idx]]) + "\n")
            df.to_csv(f, header=False, index=False)


if __name__ == "__main__":
    print('Started..')

    csv_file = 'complete_dataset_woner.csv'

    # Load the dataset
    df = pd.read_csv(csv_file)

    # Assuming 'Student' and 'Question' columns are not needed for clustering
    df_numeric = df.drop(['Student', 'Question'], axis=1)

    d = {}
    threshold_values = []
    for i in np.arange(0.01, 0.02, 0.03):
        threshold_values.append(i)
        optimizer = ClusteringAlgorithm(df_numeric, i)
        data_f = optimizer.data_f
        lbl = optimizer.labels
        error = optimizer.get_error()
        n = optimizer.no_of_clusters
        d[i] = (data_f, n, error, iter(lbl or []))

    print("Clusters made..")
    exit()

    # Save the Clusters and their centroid vectors having headers Cluster,NoOfStudentsInCluster, v1,v2,..........v_n where n is the dimension of vector
    # for k, v in d.items():
    #     data_f = v[0]
    #     labels = v[3]
    #     centroid = v[0].groupby(labels).mean()
    #     save_clusters_and_centroids(data_f, labels, centroid, f'clusters_{k}.csv')

    n_lst = [v[1] for v in d.values()]
    errors = [v[2] for v in d.values()]

    optimum_values = plot_graph(n_lst, threshold_values, errors)
    print("\nOptimum values - ", optimum_values)
    #exit the program
    actual_labels = df['Student'].tolist()  # Assuming 'Student' column contains unique identifiers

    res = {}

    # Your algorithm
    for optimum_value in optimum_values:
        data_f = d[optimum_value][0]
        n_clusters = d[optimum_value][1]
        labels = tuple(d[optimum_value][3])
        add_index(res, data_f, f"{optimum_value}", labels, actual_labels)

    # Other algorithms
    n_clusters =   5 #Number of clusters
    algorithms = {
        'Kmeans': KMeans(n_clusters=n_clusters, n_init='auto'),
        'Affinity': AffinityPropagation(),
        'Gaussian Mixture': GaussianMixture(n_components=n_clusters),
        'Agglomerative Clustering': AgglomerativeClustering(n_clusters=n_clusters)
    }

    for name, algo in algorithms.items():
        labels = algo.fit_predict(df_numeric)
        add_index(res, df_numeric, name, labels, actual_labels)

    # compare_algos(res)

# Save the Clusters and their centroid vectors


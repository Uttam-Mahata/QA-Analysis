import pandas as pd
from collections import Counter
import numpy as np

# Define the ClusteringAlgorithm class
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
        data_f = self.data_f.copy()
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
        data_f = self.data_f.copy()
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


# Load student question-answer dataset
student_dataset = pd.read_csv('complete_dataset_woner.csv')

student_dataset = student_dataset.drop(['Student', 'Question'], axis=1)

# Define threshold value for clustering
threshold_value = 0.5  # Adjust as needed

# Instantiate ClusteringAlgorithm with the dataset and threshold value
clustering_algo = ClusteringAlgorithm(student_dataset, threshold_value)

# Access the labels attribute to get cluster labels assigned to each student's answer
cluster_labels = clustering_algo.labels

# Count the number of clusters created

# Count the number of students in each cluster
student_counts_per_cluster = Counter(cluster_labels)

# Print the number of clusters created

# Print the number of students in each cluster
print("\nNumber of Students in Each Cluster:")
for cluster, count in student_counts_per_cluster.items():
    print(f"Cluster {cluster}: {count} students")

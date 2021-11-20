import pandas as pd
from sklearn.cluster import KMeans


def train_data(elbow_val, data):
    """
    Takes elbow_val and data then trains cluster and returns it
    :param elbow_val: int
    :param data: List[int]
    :return: cluster
    """
    cluster = KMeans(n_clusters=elbow_val, init='k-means++', random_state=0)
    cluster.fit(data)
    return cluster


def predict_value(cluster, data):
    """
    Takes trained cluster and data then returns prediction
    :param cluster: KMeans Cluster
    :param data: List[List[int,int,int]]
    :return: List[int]
    """
    return cluster.predict(data)


def read_data(file_url='income_spend_kmeans.csv'):
    """
    Takes file url and returns data

    :param file_url: URL
    :return: List[str]
    """
    df = pd.read_csv(file_url)
    return df.iloc[:, 2:4].values


class SpendingPredictor:

    def __init__(self, num_classes, url=None):
        """
        Initializes Prediction class with values and trains it with data
        :param num_classes: int (Elbow value)
        :param url: Any url local and web
        """
        self._num_classes = num_classes
        if url:
            self._data_url = url
        else:
            self._data_url = 'income_spend_kmeans.csv'
        self._cluster = KMeans(n_clusters=self._num_classes, init='k-means++', random_state=0)
        self.train_data(self._get_data())

    def _get_data(self):
        """
        Returns data for training from the url

        :return: List[List[int,int,int]]
        """
        df = pd.read_csv(self._data_url)
        return df.iloc[:, 2:4].values

    def train_data(self, data):
        """
        Takes data and trains with cluster

        :param data: List[List[int,int,int]]
        :return: KMeans Cluster
        """
        return self._cluster.fit(data)

    def predict_data(self, data):
        """
        Takes data(age,income) and predicts its class
        :param data: List[List[int,int]]
        :return:
        """
        return [(self._num_classes - val) for val in self._cluster.predict(data)]


if __name__ == '__main__':
    # x = read_data()
    # trained_cluster = train_data(4, x)
    # results = predict_value(trained_cluster, x)
    # print(results)
    # print(predict_value(trained_cluster, [[38, 113], [42, 86]]))
    predictor = SpendingPredictor(4, 'income_spend_kmeans.csv')
    i = 0
    while i < 5:
        age = int(input("Enter your age: "))
        income = int(input("Enter your Income: "))
        spend_prediction = predictor.predict_data([[age, income]])
        print("You are in group {}".format(spend_prediction[0]))
        i += 1

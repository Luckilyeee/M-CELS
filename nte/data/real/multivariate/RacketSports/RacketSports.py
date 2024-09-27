from sktime.datasets import load_from_tsfile
from sklearn.preprocessing import LabelEncoder
import os

#download the data .ts files from the UEA dataset website and update this data path
DATA_PATH = "MTS/DATA"

name = "RacketSports"

class RacketSportsDataset():
    def __init__(self):
        print("Loading train data . . .")
        self.train_data, self.train_label = self.load_train_data()
        print("Loading test data . . .")
        self.test_data, self.test_label = self.load_test_data()
        self.name = name

    def load_train_data(self):
        train_data, train_label = load_from_tsfile(os.path.join(DATA_PATH, name, name + "_TRAIN.ts"),
                                            return_data_type="numpy3d")
        encoder = LabelEncoder()
        train_label = encoder.fit_transform(train_label)
        TS_nums, dim_nums, ts_length = train_data.shape[0], train_data.shape[1], train_data.shape[2]

        return train_data, train_label

    def load_test_data(self):
        test_data, test_label = load_from_tsfile(os.path.join(DATA_PATH, name, name + "_TEST.ts"),
                                          return_data_type="numpy3d")

        encoder = LabelEncoder()
        test_label = encoder.fit_transform(test_label)

        return test_data, test_label


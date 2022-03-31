import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from linear_tree_shap import TreeExplainer
from shap import TreeExplainer as Truth
import pandas as pd
from sklearn.model_selection import train_test_split
import time

def dummy_transform(df):
    categorical_feature_names = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]
    for name in categorical_feature_names:
        dummy_df = pd.get_dummies(df[name])
        if "?" in dummy_df.columns.values:
            dummy_df.drop("?", axis=1, inplace=True)
        df = pd.concat([df, dummy_df], axis=1)
        df.drop(name, axis=1, inplace=True)
    return df

def load_conductor():
    data_path = '/Users/pengyu/Downloads/Fast_TreeSHAP_Code/dataset/superconduct'
    data = pd.read_csv(data_path+"/train.csv", engine = "python")
    train, test = train_test_split(data, test_size = 0.5, random_state = 0)
    label_train = train["critical_temp"]
    label_test = test["critical_temp"]
    train = train.iloc[:, :-1]
    test = test.iloc[:, :-1]
    return 'conductor', train.values.astype(np.float64), label_train, test.values.astype(np.float64)

def load_adult():
    data_path = '/Users/pengyu/Downloads/Fast_TreeSHAP_Code/dataset/adult'
    train = pd.read_csv(data_path+'/adult.data', sep = ",\s+", header = None, skiprows = 1, engine = "python")    
    test = pd.read_csv(data_path+"/adult.test", sep = ",\s+", header = None, skiprows = 1, engine = "python")
    label_train = train[14].map({"<=50K": 0, ">50K": 1}).tolist()
    label_test = test[14].map({"<=50K.": 0, ">50K.": 1}).tolist()
    train = train.iloc[:, :-2]
    test = test.iloc[:, :-2]
    feature_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
                     "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"]
    train.columns = feature_names
    test.columns = feature_names
    train = dummy_transform(train)
    test = dummy_transform(test)
    return 'adult', train.values.astype(np.float64), label_train, test.values.astype(np.float64)

def main():
    for name, train_x, train_y, test_x in [load_adult(), 
                                           load_conductor()]:
        for depth in [8, 12, 16]:
            clf = DecisionTreeRegressor(max_depth=depth).fit(train_x, train_y)
            linear = TreeExplainer(clf)
            fast = Truth(clf)
            linear_time = time.time()
            test_x = train_x
            linear_result = linear.shap_values(test_x)
            linear_time = time.time()-linear_time
            fast_time = time.time()
            fast_result = fast.shap_values(test_x)
            fast_time = time.time()-fast_time
            #print(name, 'fast', depth, fast_time)
            print(name, 'linear', depth, linear_time)
            print(depth, linear_result[0], fast_result[0])
            np.testing.assert_array_almost_equal(linear_result, fast_result, 2)


if __name__ == "__main__":
    main()

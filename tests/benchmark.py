import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
import linear_tree_shap
import fasttreeshap
import shap
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
    data_path = '~/Downloads/Fast_TreeSHAP_Code/dataset/superconduct'
    data = pd.read_csv(data_path+"/train.csv", engine = "python")
    train, test = train_test_split(data, test_size = 0.5, random_state = 0)
    label_train = train["critical_temp"]
    label_test = test["critical_temp"]
    train = train.iloc[:, :-1]
    test = test.iloc[:, :-1]
    return 'conductor', train.values.astype(np.float64), label_train, test.values.astype(np.float64)

def load_adult():
    data_path = '~/Downloads/Fast_TreeSHAP_Code/dataset/adult'
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

def methods(clf):
    return [
            ('linear_tree_shap_v1', 
             linear_tree_shap.TreeExplainer(clf), None),
            ('linear_tree_shap_v2', 
             linear_tree_shap.TreeExplainer(clf), lambda e, x: e.shap_values_v2(x)),
            ('fast_tree_shap_v1', 
            fasttreeshap.TreeExplainer(clf, algorithm='v1', n_jobs=1), None),
            ('fast_tree_shap_v2', 
            fasttreeshap.TreeExplainer(clf, algorithm='v2', n_jobs=1), None),
            ('tree_shap', 
            shap.TreeExplainer(clf), lambda e, x: e.shap_values(x, check_additivity=False)),
    ]


def main():
    print('roud,dataset,method,depth,time')
    for data_name, train_x, train_y, test_x in [load_adult(), 
                                           load_conductor()]:
        for i in range(5):
            for depth in [2, 4, 6, 8, 10, 12, 14, 16, 18]:
                clf = DecisionTreeRegressor(max_depth=depth).fit(train_x, train_y)
                for method_name, exp, executor in methods(clf):
                    if executor is None:
                        start = time.time()
                        exp.shap_values(test_x)
                        total_time = time.time()-start
                    else:
                        start = time.time()
                        executor(exp, test_x)
                        total_time = time.time()-start
                    print(','.join([str(i), data_name, method_name, str(depth), str(total_time)]))


if __name__ == "__main__":
    main()

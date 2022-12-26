import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import json

def load_data(data):
    iris = pd.read_csv(data, sep=',')
    print(iris.shape)
    return iris

def get_train_test_data(iris):

    encode = LabelEncoder()
    iris.Species = encode.fit_transform(iris.Species)

    train , test = train_test_split(iris, test_size=0.2, random_state=0)
    print('shape of training data : ', train.shape)
    print('shape of testing data', test.shape)


    X_train = train.drop(columns=['Species'], axis=1)
    y_train = train['Species']
    X_test = test.drop(columns=['Species'], axis=1)
    y_test = test['Species']

    return X_train, X_test, y_train, y_test

def evaluation(y_test,predict):
    print("accuracy : ",accuracy_score(y_test,predict))
    
    accuracy = accuracy_score(y_test,predict)
    
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'nuberValue': accuracy,
            'format': "%",
        }]
    }
    
    with open('./accuracy.json','w') as f:
        json.dump(accuracy,f)
    
    with open('./mlpipeline-metrics.json','w') as f:
        json.dump(metrics,f)


if __name__ == "__main__":
    
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument(
        '--data',
        type=str,
        default='./Iris.csv',
        help="Input data csv"
    )

    args = argument_parser.parse_args()
    iris = args.data
    iris = load_data(iris)

    X_train, X_test, y_train, y_test = get_train_test_data(iris)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print('\nAccuracy Score on test data : ')
    print(accuracy_score(y_test, predict))
    evaluation(y_test,predict)
    
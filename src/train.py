import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_model(train_path):
    # load training data
    train = pd.read_csv(train_path)

    # seprate feature(x) and target(y)
    x = train.drop("Class",axis = 1)
    y = train["Class"]

    # split data into training an validation sets
    x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.2, random_state = 42)

    # create RandomForest model
    model = RandomForestClassifier(n_estimators = 200, random_state= 42)
    
    # train model on training set
    model.fit(x_train, y_train)

    # predict on validation set
    val_pred = model.predict(x_val)

    # calculate and print accuracy score
    acc= accuracy_score(y_val, val_pred)
    print("validation Accuracy:", acc)

    #retrain model on full dataset for final use
    model.fit(x,y)

    return model

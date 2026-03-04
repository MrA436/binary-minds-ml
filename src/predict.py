import pandas as pd


def generate_predictions(model, test_path):
    # Load test data
    test = pd.read_csv(test_path)

    # Extract ID column
    test_ids = test["ID"]

    # Drop ID to get feature matrix
    x_test = test.drop("ID", axis=1)

    # Generate predictions
    predictions = model.predict(x_test)

    return test_ids, predictions
from src.train import train_model
from src.predict import generate_predictions
import pandas as pd


def main():
    model = train_model("data/TRAIN.csv")
    ids, preds = generate_predictions(model, "data/TEST.csv")

    submission = pd.DataFrame({
        "ID": ids,
        "CLASS": preds
    })

    submission.to_csv("FINAL.csv", index=False)
    print("FINAL.csv generated successfully")


if __name__ == "__main__":
    main()

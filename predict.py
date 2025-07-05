import argparse
from hate_speech.model import load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    model = load_model(args.model)
    prediction = model.predict([args.text])[0]
    label = "Hate Speech" if prediction == 1 else "Not Hate Speech"
    print(f"Prediction: {label}")
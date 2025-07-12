import json

import pandas as pd
import torch
import csv
import os
import requests
from datetime import datetime
import matplotlib

from TimeSeries import TimeSeriesDataset, TimeSeriesTrainer, TimeSeriesTransformer, device

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')

model_attributes = {
    "d_model": 128,
    "nhead": 8,
    "num_layers": 4,
    "prediction_length": 7,
    "sequence_length": 14,
    "inference_target": "distance_km",
    "test_size": 0.2,
    "validation_size": 0.1,
}


def get_weather_for_day(date, csv_file="./datafiles/weather.csv"):
    # Check if weather.csv exists and contains the date already
    date_str = date.strftime('%Y-%m-%d')

    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['date'] == date_str:
                    return {
                        "date": row['date'],
                        "midday_temp": float(row['midday_temp']),
                        "weather_category": row['weather_category'],
                        "wind_speed": float(row['wind_speed']),
                    }

    # If not found in cache, call the API
    dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = int(dt_obj.replace(hour=12).timestamp())  # Midday to avoid timezone issues

    # Brisbane coordinates
    lat = -27.4705
    lon = 153.0260

    url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
    params = {
        "lat": lat,
        "lon": lon,
        "dt": timestamp,
        "appid": "get-your-own-damn-api-key",
        "units": "metric"
    }
    print("Calling weather api for {}".format(date_str))
    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200:
        raise Exception(data.get("message", "Failed to fetch weather data."))

    midday_temp = data["data"][0]["temp"]
    wind_speed = data["data"][0]["wind_speed"]
    weather_category = data["data"][0]["weather"][0]["main"].lower()

    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["date", "midday_temp", "weather_category", "wind_speed"])
        if not file_exists:
            writer.writeheader()

        response_packet = {
            "date": date_str,
            "midday_temp": round(midday_temp, 1),
            "weather_category": weather_category,
            "wind_speed": round(wind_speed, 1),
        }
        writer.writerow(response_packet)

    return response_packet


def load_and_prepare_data(file_path):
    # load into pandas df
    df = pd.read_csv(file_path)
    target_col = model_attributes.get("inference_target")
    test_size = model_attributes.get("test_size")
    val_size = model_attributes.get("validation_size")

    # sort by datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    holidays = pd.read_csv("./datafiles/holidays.csv")
    holidays['date'] = pd.to_datetime(holidays['date'])

    # is the current day a public holiday or not?
    df['is_holiday'] = df['date'].isin(holidays['date']).astype(int)

    # for every date in the timeseries, check we have weather in the weather csv...
    df['date'].apply(get_weather_for_day)

    # weather csv should be completed now... so load it into main df.
    weather = pd.read_csv("./datafiles/weather.csv")
    weather['date'] = pd.to_datetime(weather['date'])

    # we have a bias on weather that we want to pass down to the transformer.
    def weather_category_score(weather_string):
        scores = {
            "fog": -30,
            "rain": -50,
            "clouds": 0,
            "clear": 50
        }
        return scores.get(weather_string, 0)

    df['midday_temp'] = weather['midday_temp']
    df['weather_category'] = weather['weather_category'].apply(weather_category_score)
    df['wind_speed'] = weather['wind_speed']

    df.to_csv('./datafiles/merged-dataset.csv', index=False)

    #ignore the last prediction length rows of data since we'll be using that for inference ongoing...
    df = df[:-model_attributes.get("prediction_length")]

    # all columns should be features except date and target.
    feature_cols = [col for col in df.columns if col not in ['date', target_col]]

    # split data into training, testing and validation sets.
    n = len(df)
    train_end = int(n * (1 - test_size - val_size))
    val_end = int(n * (1 - test_size))

    train_data = df[:train_end]
    val_data = df[train_end:val_end]
    test_data = df[val_end:]

    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")

    sequence_length = model_attributes.get("sequence_length")
    prediction_length = model_attributes.get("prediction_length")

    train_dataset = TimeSeriesDataset(train_data, target_col, feature_cols, sequence_length, prediction_length)
    val_dataset = TimeSeriesDataset(val_data, target_col, feature_cols, sequence_length, prediction_length)
    test_dataset = TimeSeriesDataset(test_data, target_col, feature_cols, sequence_length, prediction_length)

    return train_dataset, val_dataset, test_dataset, feature_cols


def plot_predictions(predictions, actuals, dataset, num_samples=5):
    try:
        plt.figure(figsize=(15, 10))

        for i in range(min(num_samples, len(predictions))):
            plt.subplot(num_samples, 1, i + 1)

            # Convert back to original scale
            actual_original = dataset.target_scaler.inverse_transform(actuals[i].reshape(-1, 1)).flatten()
            pred_original = dataset.target_scaler.inverse_transform(predictions[i].reshape(-1, 1)).flatten()
            plt.plot(actual_original, label='Actual', marker='o')
            plt.plot(pred_original, label='Predicted', marker='s')
            plt.title(f'Sample {i + 1}: Kms Forecast')
            plt.xlabel('Days')
            plt.ylabel('Distance Driven (Km)')
            plt.legend()
            plt.ylim(bottom=0)
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('./models/current/performance.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print("Predictions plot saved as './models/current/performance.png'")
    except Exception as e:
        print(f"Could not create predictions plot: {e}")


def save_model_artifacts(model, train_dataset, val_dataset, test_dataset, feature_cols, metrics):
    config = {
        'input_dim': len(feature_cols) + 1,
        'd_model': model_attributes.get('d_model'),
        'nhead': model_attributes.get('nhead'),
        'num_layers': model_attributes.get('num_layers'),
        'prediction_length': model_attributes.get('prediction_length'),
        'sequence_length': model_attributes.get('sequence_length'),
        'inference_target': model_attributes.get('inference_target'),
        'feature_cols': feature_cols,
        'training_metrics': metrics,
        'created_at': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'model_type': 'TimeSeriesTransformer'
    }

    with open('./models/current/model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Model config saved to './models/current/model_config.json''")

    scaler_data = {
        'feature_scaler': train_dataset.feature_scaler,
        'target_scaler': train_dataset.target_scaler,
        'feature_cols': feature_cols,
        'target_col': model_attributes.get("inference_target")
    }

    torch.save(scaler_data, './models/current/model_scalers.pth')

    print("python inference.py")


def main():
    # Load and prepare data
    train_dataset, val_dataset, test_dataset, feature_cols = load_and_prepare_data(
        'datafiles/distance-driven.csv'
    )

    # include the history of the target as a feature.
    input_dim = len(feature_cols) + 1  # features + target history

    # model_attributes
    d_model = model_attributes.get("d_model")
    nhead = model_attributes.get("nhead")
    num_layers = model_attributes.get("num_layers")
    prediction_length = model_attributes.get("prediction_length")

    print(f"Model input dimension: {input_dim}")
    print(f"Features: {feature_cols}")

    # use the HF transformer model
    model = TimeSeriesTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        prediction_length=prediction_length
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # train
    trainer = TimeSeriesTrainer(model, train_dataset, val_dataset, test_dataset)
    trainer.train()

    # eval
    predictions, actuals, metrics = trainer.evaluate()

    # plot
    plot_predictions(predictions, actuals, test_dataset)

    # save
    torch.save(model.state_dict(), './models/current/timeseries_model.pth')
    print("Model saved as './models/current/timeseries_model.pth'")

    return model, predictions, actuals, metrics, feature_cols


def predict_future(model, dataset, last_sequence_idx=-1):
    # toggle to eval mode
    model.eval()

    # last sequence from dataset
    last_sequence = dataset.sequences[last_sequence_idx]

    with torch.no_grad():
        # unsqueeze - add fake batch of size 1 since NN expects a batch number...
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
        outputs = model(input_tensor)
        predictions = outputs['logits'].cpu().numpy()  # Use 'logits' key

    # Convert back to original scale
    predictions_original = dataset.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions_original


if __name__ == "__main__":
    # Run the complete pipeline
    model, predictions, actuals, metrics, feature_cols = main()

    # Example of future prediction
    train_dataset, val_dataset, test_dataset, _ = load_and_prepare_data('./datafiles/distance-driven.csv')
    future_predictions = predict_future(model, test_dataset)
    save_model_artifacts(model, train_dataset, val_dataset, test_dataset, feature_cols, metrics)

    print(
        f"Utes will travel a combined total of {sum(future_predictions)}kms. over the next {len(future_predictions)} days.")

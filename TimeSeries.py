
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


matplotlib.use('Agg')


class TimeSeriesDataset(Dataset):
    def __init__(self, data, target_col, feature_cols, sequence_length=0, prediction_length=0):
        # full dataset
        self.data = data

        # what column are we predicting?
        self.target_col = target_col

        # what are the features to this dataset?
        self.feature_cols = feature_cols

        # context window
        self.sequence_length = sequence_length

        # future looking period
        self.prediction_length = prediction_length

        # scalers
        self.target_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

        self.scaled_target = self.target_scaler.fit_transform(data[[target_col]])
        self.scaled_features = self.feature_scaler.fit_transform(data[feature_cols])

        # sequences
        self.sequences = []
        self.targets = []

        # for each "sliding window" of data we want to predict (all data except the last periods we want to predict)
        for observation in range(len(data) - sequence_length - prediction_length + 1):
            # get the slice of data we're looking at (features + target history)
            seq_features = self.scaled_features[observation:(observation + sequence_length)]
            seq_target = self.scaled_target[observation:(observation + sequence_length)]

            # merge the target with features for full sequence input
            seq_input = np.concatenate([seq_features, seq_target], axis=1)

            # the target sequence we want to predict (prediction length after the current sequence length)
            target_seq = self.scaled_target[
                         (observation + sequence_length):(observation + sequence_length + prediction_length)]

            self.sequences.append(seq_input)
            self.targets.append(target_seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # need tensors to track gradients for back-prop.
        return {
            'input_ids': torch.FloatTensor(self.sequences[idx]),
            'labels': torch.FloatTensor(self.targets[idx])
        }

# Custom transformer for time series data
class TimeSeriesTransformer(nn.Module):

    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, prediction_length=0):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.prediction_length = prediction_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, prediction_length)
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, labels=None):
        batch_size, seq_length, _ = input_ids.shape

        # Project input
        x = self.input_projection(input_ids)

        # Add positional encoding
        x = x + self.positional_encoding[:seq_length].unsqueeze(0)

        # Apply layer norm
        x = self.layer_norm(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output projection
        logits = self.output_projection(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits, labels.squeeze(-1))

        # Return a simple dict that Trainer can handle
        return {
            'loss': loss,
            'logits': logits
        }

class TimeSeriesTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Training arguments
        self.training_args = TrainingArguments(
            # Where to save hugging face checkpoints
            output_dir='./results',
            # how many passes through the data set
            num_train_epochs=50,
            #num_train_epochs=10,

            # batch 32 samples at once.
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,

            # less volatile gradient updates early on.
            # training data set is about 800 samples
            # 800/batch size(32) = 25 steps per epoch
            # 25 steps per epoch ->  20 epochs for warm up
            warmup_steps=500,

            # L2 regularization coefficient adds penalty to loss: loss = mse_loss + 0.01 * sum(param^2)
            weight_decay=0.01,

            logging_dir='./logs',
            logging_steps=10,

            eval_strategy="epoch",
            save_strategy="epoch",

            load_best_model_at_end=True,

            # objective is to minimise loss factor
            metric_for_best_model="eval_loss",

            # less loss is better (not higher)
            greater_is_better=False,

            save_total_limit=3,

            learning_rate=1e-4,

            # For more aggressive training:
            # learning_rate=5e-4
            # weight_decay = 0.1
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

    def train(self):
        print("Starting training...")
        self.trainer.train()
        print("Training completed!")

    def evaluate(self):
        print("Evaluating on test set...")
        test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False, pin_memory=False)

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids)
                pred = outputs['logits']  # Use 'logits' key instead of 'last_hidden_state'

                predictions.extend(pred.cpu().numpy())
                actuals.extend(labels.squeeze(-1).cpu().numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
        mse = mean_squared_error(actuals.flatten(), predictions.flatten())
        rmse = np.sqrt(mse)

        print(f"Test Results:")
        print(f"MAE: {mae:.4f}")  # Mean Absolute Error
        print(f"MSE: {mse:.4f}")  # Mean Squared Error
        print(f"RMSE: {rmse:.4f}")  # Root Mean Squared Error (√MSE)

        return predictions, actuals, {'mae': mae, 'mse': mse, 'rmse': rmse}
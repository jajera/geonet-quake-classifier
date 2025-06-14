"""
GeoNet Quake Classifier - Transformer Model

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using a lightweight transformer model (DistilBERT)
for attention-based classification.
It uses the utilities module to reduce code duplication.
"""

import warnings
from config import (
    TRANSFORMER_MODEL_MAP_INTENSITY_TYPE,
    MIN_MAGNITUDE,
    TRANSFORMER_MODEL_CONFIG,
)

import numpy as np
import utilities

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
    from transformers import logging as transformers_logging
    import torch
    from torch.utils.data import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Reduce transformers logging
    transformers_logging.set_verbosity_error()
    TRANSFORMERS_AVAILABLE = True
except (ImportError, ValueError) as e:
    print(
        "⚠️  Transformers library not available. "
        "Installing fallback classifier..."  # noqa: E501
    )  # noqa: E501
    if "Keras" in str(e):
        print("   (Keras 3 compatibility issue detected)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Define a dummy Dataset class for fallback
    class Dataset:
        pass

    TRANSFORMERS_AVAILABLE = False


class EarthquakeDataset(Dataset):
    """Custom dataset for earthquake text data."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class QuakeClassifierTransformer:
    """
    A class to handle earthquake data processing and classification
    using Transformer models (DistilBERT).

    This class provides methods to fetch earthquake data from GeoNet,
    process and filter the data, train a transformer classifier,
    and generate map visualization data.
    """

    def __init__(self, filters=None):
        """
        Initialize the QuakeClassifierTransformer with optional filters.

        Args:
            filters (dict, optional): Custom filters for the GeoNet API query.
                Defaults to DEFAULT_FILTERS from config.
        """
        self.filters = filters
        self.api_url = utilities.build_api_url(filters)

        if TRANSFORMERS_AVAILABLE:
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                TRANSFORMER_MODEL_CONFIG["model_name"]
            )
            self.model = None  # Will be initialized during training
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000)
            self.model = LogisticRegression(random_state=42)

    def text_format_data(self, df):
        """
        Convert earthquake data to text format for transformer processing.

        Args:
            df (pd.DataFrame): Earthquake data.

        Returns:
            list: List of text representations of earthquake data.
        """
        texts = []
        for _, row in df.iterrows():
            # Convert numerical data to descriptive text
            magnitude_desc = f"magnitude {row['magnitude']:.1f}"
            depth_desc = f"depth {row['depth']:.1f} kilometers"
            text = f"Earthquake with {magnitude_desc} at {depth_desc}"
            texts.append(text)
        return texts

    def train_model(self, texts, labels):
        """
        Train the transformer model.

        Args:
            texts (list): List of text representations.
            labels (list): List of intensity labels.

        Returns:
            float: Model accuracy score.
        """
        if TRANSFORMERS_AVAILABLE:
            return self._train_transformer_model(texts, labels)
        else:
            return self._train_fallback_model(texts, labels)

    def _train_transformer_model(self, texts, labels):
        """Train using DistilBERT transformer."""
        print("\nTraining DistilBERT Transformer Model...")

        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Create datasets
        train_dataset = EarthquakeDataset(
            train_texts,
            train_labels,
            self.tokenizer,
            TRANSFORMER_MODEL_CONFIG["max_length"],
        )
        test_dataset = EarthquakeDataset(
            test_texts,
            test_labels,
            self.tokenizer,
            TRANSFORMER_MODEL_CONFIG["max_length"],
        )

        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            TRANSFORMER_MODEL_CONFIG["model_name"],
            num_labels=TRANSFORMER_MODEL_CONFIG["num_labels"],
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=TRANSFORMER_MODEL_CONFIG["num_epochs"],
            per_device_train_batch_size=TRANSFORMER_MODEL_CONFIG["batch_size"],
            per_device_eval_batch_size=TRANSFORMER_MODEL_CONFIG["batch_size"],
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # Train model
        trainer.train()

        # Evaluate model
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        accuracy = accuracy_score(test_labels, pred_labels)

        print("\nDistilBERT Transformer Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                test_labels, pred_labels, target_names=["Low", "High"]
            )
        )

        return accuracy

    def _train_fallback_model(self, texts, labels):
        """Train using TF-IDF + Logistic Regression fallback."""
        print("\nTraining TF-IDF + Logistic Regression Fallback Model...")

        # Vectorize text data
        X = self.vectorizer.fit_transform(texts)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)

        # Evaluate model
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print("\nTF-IDF Fallback Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, predictions, target_names=["Low", "High"]
            )  # noqa: E501
        )  # noqa: E501

        return accuracy

    def predict_intensity(self, text):
        """
        Predict earthquake intensity using the trained model.

        Args:
            text (str): Text representation of earthquake data.

        Returns:
            dict: Predicted intensity class and label.
        """
        if TRANSFORMERS_AVAILABLE and self.model:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=TRANSFORMER_MODEL_CONFIG["max_length"],
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                )  # noqa: E501
                pred = torch.argmax(predictions, dim=-1).item()
        else:
            # Fallback prediction
            text_vec = self.vectorizer.transform([text])
            pred = self.model.predict(text_vec)[0]

        return {
            "intensity_class": pred,
            "intensity": "High" if pred else "Low",
        }


def main():
    """
    Main function to run the earthquake classification and mapping process.
    """
    model_type = "DistilBERT" if TRANSFORMERS_AVAILABLE else "TF-IDF Fallback"
    print(f"🤖 GeoNet Earthquake Transformer Model Classifier ({model_type})")
    print("=" * 60)
    print("Fetching earthquake data from GeoNet...")

    qc = QuakeClassifierTransformer()
    features = utilities.fetch_data(qc.api_url)
    if not features:
        print("No features fetched.")
        return

    df = utilities.process_data(features)
    if df.empty:
        print("No data after processing.")
        return

    df = utilities.filter_data(df)
    if df.empty:
        print("No data after filtering.")
        return

    # Calculate intensity_class for training
    df["intensity_class"] = df["magnitude"].apply(
        utilities.calculate_intensity_class
    )  # noqa: E501

    # Print dataset summary
    utilities.print_dataset_summary(df)

    # Convert to text format
    texts = qc.text_format_data(df)
    labels = df["intensity_class"].tolist()

    # Determine the final intensity for mapping based on config
    if TRANSFORMER_MODEL_MAP_INTENSITY_TYPE == "predicted":
        # Ensure there's enough data to train
        if len(df) < 2:
            print(
                "Not enough data to train the model. "
                "Using threshold-based classification."
            )
            df = utilities.apply_intensity_labels(df)
            acc = 0.0
        else:
            # Train the model and get accuracy
            acc = qc.train_model(texts, labels)

            # Predict intensity for all data
            predictions = []
            for text in texts:
                pred_result = qc.predict_intensity(text)
                predictions.append(pred_result["intensity"])

            df["intensity"] = predictions
    else:
        # Use the calculated intensity_class as the actual intensity
        df = utilities.apply_intensity_labels(df)
        acc = 0.0

    print("\nGenerating interactive map...")
    map_data = utilities.generate_map_data(df)
    if not map_data:
        print("No map data generated.")
        return

    # Generate HTML using utilities
    model_config = {
        "name": "Transformer Model",
        "emoji": "🤖",
        "header_color": "#6b46c1",
    }
    html_content = utilities.generate_map_html(
        map_data,
        MIN_MAGNITUDE,
        model_config,
        TRANSFORMER_MODEL_MAP_INTENSITY_TYPE,
    )
    utilities.save_html_file(html_content, "transformer_model.html")

    print("\n✅ Classification complete!")
    if acc > 0:
        print(f"Model accuracy: {acc:.4f}")
    print(f"Processed {len(df)} earthquakes")
    print("🤖 View results: transformer_model.html")


if __name__ == "__main__":
    main()

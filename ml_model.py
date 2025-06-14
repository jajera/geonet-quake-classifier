"""
GeoNet Quake Classifier - Machine Learning Model (Refactored)

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using a focused Machine Learning approach with Logistic Regression.
It uses the utilities module to reduce code duplication.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from config import (
    ML_MODEL_MAP_INTENSITY_TYPE,
    ML_MODEL_CONFIG,
    MIN_MAGNITUDE,
)

import utilities


class QuakeClassifierML:
    """
    A class to handle earthquake data processing and classification
    using Machine Learning (Logistic Regression).

    This class provides methods to fetch earthquake data from GeoNet,
    process and filter the data, train a logistic regression classifier,
    and generate map visualization data.
    """

    def __init__(self, filters=None):
        """
        Initialize the QuakeClassifierML with optional filters.

        Args:
            filters (dict, optional): Custom filters for the GeoNet API query.
                Defaults to DEFAULT_FILTERS from config.
        """
        self.filters = filters
        # Filter out model_type from config for LogisticRegression
        lr_params = {
            k: v for k, v in ML_MODEL_CONFIG.items() if k != "model_type"
        }  # noqa: E501
        self.model = LogisticRegression(**lr_params)
        self.scaler = StandardScaler()
        self.api_url = utilities.build_api_url(filters)

    def predict_intensity(self, magnitude, depth):
        """
        Predict earthquake intensity using the trained model.

        Args:
            magnitude (float): Earthquake magnitude.
            depth (float): Earthquake depth in kilometers.

        Returns:
            dict: Predicted intensity class and label.
        """
        scaled_input = self.scaler.transform([[magnitude, depth]])
        pred = self.model.predict(scaled_input)[0]
        return {
            "intensity_class": pred,
            "intensity": "High" if pred else "Low",
        }

    def train_model(
        self,
        features_train,
        labels_train,
        features_test,
        labels_test,
    ):
        """
        Train the machine learning model and generate performance metrics.

        Args:
            features_train (pd.DataFrame): Training features.
            labels_train (pd.Series): Training labels.
            features_test (pd.DataFrame): Test features.
            labels_test (pd.Series): Test labels.

        Returns:
            float: Model accuracy score.
        """
        # Scale features
        features_train_scaled = self.scaler.fit_transform(features_train)
        features_test_scaled = self.scaler.transform(features_test)

        # Train the model
        self.model.fit(features_train_scaled, labels_train)

        # Evaluate accuracy
        predictions = self.model.predict(features_test_scaled)
        accuracy = accuracy_score(labels_test, predictions)

        # Print model performance
        print("\nLogistic Regression ML Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(
            classification_report(
                labels_test, predictions, target_names=["Low", "High"]
            )
        )

        # Generate model visualization
        self._generate_model_visualization(features_train, labels_train)

        return accuracy

    def _generate_model_visualization(self, features_train, labels_train):
        """
        Generate and save model visualization as PNG.

        Args:
            features_train (pd.DataFrame): Training features.
            labels_train (pd.Series): Training labels.
        """
        plt.figure(figsize=(12, 8))

        # Create a decision boundary plot
        h = 0.02  # step size in the mesh

        # Get feature ranges
        x_min = features_train.iloc[:, 0].min() - 1
        x_max = features_train.iloc[:, 0].max() + 1
        y_min = features_train.iloc[:, 1].min() - 1
        y_max = features_train.iloc[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
        )  # noqa: E501

        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        mesh_points_scaled = self.scaler.transform(mesh_points)
        Z = self.model.predict(mesh_points_scaled)
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlGn)

        # Plot the training points
        scatter = plt.scatter(
            features_train.iloc[:, 0],
            features_train.iloc[:, 1],
            c=labels_train,
            cmap=plt.cm.RdYlGn,
            edgecolors="black",
        )

        # Add labels and title
        plt.xlabel("Magnitude")
        plt.ylabel("Depth (km)")
        plt.title(
            "Machine Learning Model (Logistic Regression)\n"
            "Decision Boundary and Training Data"
        )

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Intensity Class")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig("ml_model.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("Model visualization saved as 'ml_model.png'")


def main():
    """
    Main function to run the earthquake classification and mapping process.
    """
    print("🤖 GeoNet Earthquake ML Model Classifier")
    print("=" * 48)
    print("Fetching earthquake data from GeoNet...")

    qc = QuakeClassifierML()
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

    # Determine the final intensity for mapping based on config
    if ML_MODEL_MAP_INTENSITY_TYPE == "predicted":
        # Prepare data for training
        features = df[["magnitude", "depth"]]
        labels = df["intensity_class"]

        # Ensure there's enough data to split and train
        if len(df) < 2:
            print("Not enough data to train the model. Skipping prediction.")
            df = utilities.apply_intensity_labels(df)
            acc = 0.0
        else:
            print("\nTraining logistic_regression model...")

            # Split data for training and testing
            # fmt: off
            features_train, features_test, labels_train, labels_test = (
                train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
            )
            # fmt: on

            # Train the model and get accuracy
            acc = qc.train_model(
                features_train, labels_train, features_test, labels_test
            )

            # Predict intensity using the trained model
            # on the entire filtered data
            df["intensity"] = qc.model.predict(
                qc.scaler.transform(df[["magnitude", "depth"]])
            )
            df["intensity"] = df["intensity"].apply(
                lambda x: "High" if x else "Low"
            )  # noqa: E501
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
        "name": "ML Model",
        "emoji": "🤖",
        "header_color": "#8b5a3c",
    }
    html_content = utilities.generate_map_html(
        map_data,
        MIN_MAGNITUDE,
        model_config,
        ML_MODEL_MAP_INTENSITY_TYPE,
    )
    utilities.save_html_file(html_content, "ml_model.html")

    print("\n✅ Classification complete!")
    if acc > 0:
        print(f"Model accuracy: {acc:.4f}")
    print(f"Processed {len(df)} earthquakes")
    print("🤖 View results: ml_model.html")


if __name__ == "__main__":
    main()

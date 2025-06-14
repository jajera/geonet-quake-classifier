"""
GeoNet Quake Classifier - Statistical Model

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using statistical models (Naive Bayes or Logistic Regression).
It uses the utilities module to reduce code duplication.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from config import (
    STATISTICAL_MODEL_MAP_INTENSITY_TYPE,
    MIN_MAGNITUDE,
    STATISTICAL_MODEL_CONFIG,
)

import utilities


class QuakeClassifierStatistical:
    """
    A class to handle earthquake data processing and classification
    using statistical models.

    This class provides methods to fetch earthquake data from GeoNet,
    process and filter the data, train a statistical classifier,
    and generate map visualization data.
    """

    def __init__(self, filters=None):
        """
        Initialize the QuakeClassifierStatistical with optional filters.

        Args:
            filters (dict, optional): Custom filters for the GeoNet API query.
                Defaults to DEFAULT_FILTERS from config.
        """
        self.filters = filters
        self.model = self._build_model()
        self.scaler = StandardScaler()
        self.api_url = utilities.build_api_url(filters)

    def _build_model(self):
        """
        Build the statistical model based on configuration.

        Returns:
            sklearn model: The configured statistical model.
        """
        model_type = STATISTICAL_MODEL_CONFIG["model_type"]

        if model_type == "logistic_regression":
            config = STATISTICAL_MODEL_CONFIG["logistic_regression"]
            return LogisticRegression(**config)
        elif model_type == "naive_bayes":
            config = STATISTICAL_MODEL_CONFIG["naive_bayes"]
            return GaussianNB(**config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def predict_intensity(self, magnitude, depth):
        """
        Predict earthquake intensity using the trained model.

        Args:
            magnitude (float): Earthquake magnitude.
            depth (float): Earthquake depth in kilometers.

        Returns:
            dict: Predicted intensity class and label.
        """
        # Scale the input for logistic regression
        if STATISTICAL_MODEL_CONFIG["model_type"] == "logistic_regression":
            scaled_input = self.scaler.transform([[magnitude, depth]])
            pred = self.model.predict(scaled_input)[0]
        else:
            # Naive Bayes doesn't require scaling
            pred = self.model.predict([[magnitude, depth]])[0]

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
        Train the statistical model and generate performance metrics.

        Args:
            features_train (pd.DataFrame): Training features.
            labels_train (pd.Series): Training labels.
            features_test (pd.DataFrame): Test features.
            labels_test (pd.Series): Test labels.

        Returns:
            float: Model accuracy score.
        """
        # Scale features for logistic regression
        if STATISTICAL_MODEL_CONFIG["model_type"] == "logistic_regression":
            features_train_scaled = self.scaler.fit_transform(features_train)
            features_test_scaled = self.scaler.transform(features_test)
        else:
            # Naive Bayes doesn't require scaling
            features_train_scaled = features_train
            features_test_scaled = features_test

        # Train the model
        self.model.fit(features_train_scaled, labels_train)

        # Evaluate accuracy
        predictions = self.model.predict(features_test_scaled)
        accuracy = accuracy_score(labels_test, predictions)

        # Print model performance
        model_name = (
            STATISTICAL_MODEL_CONFIG["model_type"].replace("_", " ").title()
        )  # noqa: E501
        print(f"\n{model_name} Model Performance:")
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
        if STATISTICAL_MODEL_CONFIG["model_type"] == "logistic_regression":
            mesh_points_scaled = self.scaler.transform(mesh_points)
            Z = self.model.predict(mesh_points_scaled)
        else:
            Z = self.model.predict(mesh_points)

        Z = Z.reshape(xx.shape)

        # Plot the decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

        # Plot the training points
        scatter = plt.scatter(
            features_train.iloc[:, 0],
            features_train.iloc[:, 1],
            c=labels_train,
            cmap=plt.cm.RdYlBu,
            edgecolors="black",
        )

        # Add labels and title
        plt.xlabel("Magnitude")
        plt.ylabel("Depth (km)")
        model_name = (
            STATISTICAL_MODEL_CONFIG["model_type"].replace("_", " ").title()
        )  # noqa: E501
        plt.title(
            f"{model_name} Classification\nDecision Boundary and Training Data"
        )  # noqa: E501

        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label("Intensity Class")
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"])

        # Add grid
        plt.grid(True, alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig("statistical_model.png", dpi=300, bbox_inches="tight")
        plt.close()

        print("Model visualization saved as 'statistical_model.png'")


def main():
    """
    Main function to run the complete earthquake classification pipeline.
    """
    print("🌍 GeoNet Earthquake Statistical Model Classifier")
    print("=" * 50)

    # Initialize classifier
    classifier = QuakeClassifierStatistical()

    # Fetch data
    print("Fetching earthquake data from GeoNet...")
    features = utilities.fetch_data(classifier.api_url)

    if not features:
        print("No data fetched. Exiting.")
        return

    # Process data
    df = utilities.process_data(features)
    filtered_df = utilities.filter_data(df)

    if filtered_df.empty:
        print("No earthquakes found after filtering. Exiting.")
        return

    # Prepare features and labels
    features = filtered_df[["magnitude", "depth"]]
    labels = filtered_df["magnitude"].apply(
        utilities.calculate_intensity_class
    )  # noqa: E501

    # Print dataset summary
    filtered_df["intensity_class"] = labels
    utilities.print_dataset_summary(filtered_df)

    # Train-test split
    # fmt: off
    features_train, features_test, labels_train, labels_test = (
        train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    )  # noqa: E501
    # fmt: on

    # Train model
    print(f"\nTraining {STATISTICAL_MODEL_CONFIG['model_type']} model...")
    accuracy = classifier.train_model(
        features_train, labels_train, features_test, labels_test
    )

    # Generate predictions for all data
    predictions = []
    for _, row in filtered_df.iterrows():
        pred = classifier.predict_intensity(row["magnitude"], row["depth"])
        predictions.append(pred["intensity"])

    # Add predictions to dataframe
    filtered_df = filtered_df.copy()

    if STATISTICAL_MODEL_MAP_INTENSITY_TYPE == "predicted":
        filtered_df["intensity"] = predictions
    else:
        # Use actual intensity based on magnitude threshold
        filtered_df = utilities.apply_intensity_labels(filtered_df)

    # Generate map data and HTML
    print("\nGenerating interactive map...")
    map_data = utilities.generate_map_data(filtered_df)

    # Generate HTML using utilities
    model_name = (
        STATISTICAL_MODEL_CONFIG["model_type"].replace("_", " ").title()
    )  # noqa: E501
    model_config = {
        "name": f"Statistical Model ({model_name})",
        "emoji": "📊",
        "header_color": "#2c3e50",
    }
    html_content = utilities.generate_map_html(
        map_data,
        MIN_MAGNITUDE,
        model_config,
    )
    utilities.save_html_file(html_content, "statistical_model.html")

    print("\n✅ Classification complete!")
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Processed {len(filtered_df)} earthquakes")
    print("📊 View results: statistical_model.html")


if __name__ == "__main__":
    main()

"""
GeoNet Quake Classifier - Decision Tree Model (Refactored)

This module provides functionality to fetch, process,
and classify earthquake data from GeoNet New Zealand's API
using a Decision Tree classifier. It uses the utilities module
to reduce code duplication.
"""

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

from config import (
    DECISION_TREE_MAP_INTENSITY_TYPE,
    MIN_MAGNITUDE,
)

import utilities


class QuakeClassifierDecisionTree:
    """
    A class to handle earthquake data processing and classification.

    This class provides methods to fetch earthquake data from GeoNet,
    process and filter the data, train a decision tree classifier,
    and generate map visualization data.
    """

    def __init__(self, filters=None):
        """
        Initialize the QuakeClassifierDecisionTree with optional filters.

        Args:
            filters (dict, optional): Custom filters for the GeoNet API query.
                Defaults to DEFAULT_FILTERS from config.
        """
        self.filters = filters
        self.model = DecisionTreeClassifier(random_state=42)
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
        Train the decision tree classifier and generate visualization.

        Args:
            features_train (pd.DataFrame): Training features.
            labels_train (pd.Series): Training labels.
            features_test (pd.DataFrame): Test features.
            labels_test (pd.Series): Test labels.

        Returns:
            float: Model accuracy score.
        """
        # Train the model using the provided split data
        self.model.fit(features_train, labels_train)

        # Evaluate accuracy
        accuracy = accuracy_score(
            labels_test, self.model.predict(features_test)
        )  # noqa: E501

        # Plot the decision tree
        plt.figure(figsize=(10, 6))
        plot_tree(
            self.model,
            filled=True,
            feature_names=["Magnitude", "Depth"],
            class_names=["Low", "High"],
        )
        plt.savefig("decision_tree_model.png")
        plt.close()

        return accuracy


def main():
    """
    Main function to run the earthquake classification and mapping process.
    """
    print("🌳 GeoNet Earthquake Decision Tree Classifier")
    print("=" * 45)
    print("Fetching earthquake data from GeoNet...")

    qc = QuakeClassifierDecisionTree()
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
    if DECISION_TREE_MAP_INTENSITY_TYPE == "predicted":
        # Prepare data for training
        features = df[["magnitude", "depth"]]
        labels = df["intensity_class"]

        # Ensure there's enough data to split and train
        if len(df) < 2:
            print(
                "Not enough data to train the model. "
                "Using threshold-based classification."
            )
            df = utilities.apply_intensity_labels(df)
            acc = 0.0
        else:
            print("\nTraining decision tree classifier...")

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
            df["intensity"] = qc.model.predict(df[["magnitude", "depth"]])
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
        "name": "Decision Tree Model",
        "emoji": "🌳",
        "header_color": "#2c5282",
    }
    html_content = utilities.generate_map_html(
        map_data, MIN_MAGNITUDE, model_config
    )
    utilities.save_html_file(html_content, "decision_tree.html")

    print("\n✅ Classification complete!")
    if acc > 0:
        print(f"Model accuracy: {acc:.4f}")
    print(f"Processed {len(df)} earthquakes")
    print("🌳 View results: decision_tree.html")


if __name__ == "__main__":
    main()

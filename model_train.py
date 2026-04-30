import os
import json
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
import warnings

# Suppress sklearn convergence warnings for cleaner output
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def main():
    # Paths
    artifacts_dir = "artifacts_v2"
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    # Load data
    features_path = os.path.join(artifacts_dir, "features.npy")
    labels_path = os.path.join(artifacts_dir, "labels.npy")
    video_ids_path = os.path.join(artifacts_dir, "video_ids.json")
    splits_path = os.path.join(artifacts_dir, "splits.json")

    print("Loading data...")
    features = np.load(features_path)
    labels = np.load(labels_path)

    with open(video_ids_path, "r") as f:
        video_ids = json.load(f)

    with open(splits_path, "r") as f:
        splits = json.load(f)

    # Map video_ids to their index
    vid_to_idx = {vid: idx for idx, vid in enumerate(video_ids)}

    # Helper to get indices for a split
    def get_indices(split_vids):
        # some vids might not be in our features if they failed extraction
        return [vid_to_idx[vid] for vid in split_vids if vid in vid_to_idx]

    train_idx = get_indices(splits.get("train", []))
    val_idx = get_indices(splits.get("val", []))
    test_idx = get_indices(splits.get("test", []))

    print(
        f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
    )

    if len(train_idx) == 0:
        print("Error: No training data found.")
        return

    X_train, y_train = features[train_idx], labels[train_idx]
    X_val, y_val = features[val_idx], labels[val_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if len(test_idx) > 0:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = []

    # Define models
    # Note: the models are instantiated as Classifiers, which resolves the sklearn regression warning.
    models = {
        "SVM": SVC(kernel="rbf", C=30.0, gamma="scale"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=16,  # LIMIT DEPTH
            min_samples_leaf=3,  # PREVENT MEMORIZATION
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=0.01,  # increase regularization
            max_iter=300,
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }

    best_model_name = None
    best_model = None
    best_val_acc = -1.0

    print("\nTraining and Evaluating models...")
    for name, model in models.items():
        print(f"\n--- {name} ---")
        model.fit(X_train_scaled, y_train)

        # Train Accuracy
        y_train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        print(f"Train Accuracy:      {train_acc:.4f}")

        # Validation Accuracy
        if len(val_idx) > 0:
            y_val_pred = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_name = name
                best_model = model
        else:
            print("No validation split found.")
            best_model_name = name
            best_model = model

    if len(val_idx) > 0:
        print(f"\n=========================================")
        print(f"Best Model Selected: {best_model_name} (Val Acc: {best_val_acc:.4f})")
        print(f"=========================================")

    # Evaluate best model on test set
    if len(test_idx) > 0 and best_model is not None:
        print(f"\nEvaluating Best Model ({best_model_name}) on Test Set...")
        y_test_pred = best_model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_acc:.4f}")

        # print("\nClassification Report (Test Set Top Metrics):")
        # # limit output to avoid clutter, using zero_division to handle classes with no predicted samples
        # print(classification_report(y_test, y_test_pred, zero_division=0))
    elif len(test_idx) == 0:
        print("\nNo test split available for final evaluation.")

    # Save best model and scaler
    if best_model is not None:
        model_save_path = os.path.join(
            models_dir, f"best_model_{best_model_name.replace(' ', '_').lower()}.pkl"
        )
        scaler_save_path = os.path.join(models_dir, "scaler.pkl")

        print(f"\nSaving {best_model_name} model to {model_save_path}")
        joblib.dump(best_model, model_save_path)
        print(f"Saving scaler to {scaler_save_path}")
        joblib.dump(scaler, scaler_save_path)

    print("Done!")


if __name__ == "__main__":
    main()

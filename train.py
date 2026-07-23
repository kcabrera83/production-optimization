"""Train field optimizer and allocation model, save artifacts."""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

from production_optimization.data_generator import generate_field_data
from production_optimization.utils.preprocessor import (
    FieldPreprocessor, OPT_TARGET, ALLOC_TARGET,
)
from production_optimization.models.field_optimizer import FieldOptimizer
from production_optimization.models.allocation_model import AllocationModel


def main():
    os.makedirs("outputs/models", exist_ok=True)

    print("Generating synthetic field data ...")
    df = generate_field_data(n_samples=2000, random_state=42)
    df.to_csv("outputs/field_data.csv", index=False)
    print(f"  Dataset shape: {df.shape}")

    # --- Field Optimizer (net_profit) ---
    print("\nTraining Field Optimizer (GradientBoosting) ...")
    preproc_opt = FieldPreprocessor()
    X_opt, y_opt = preproc_opt.fit_transform(df, OPT_TARGET)
    feature_names = preproc_opt.get_feature_names()

    optimizer = FieldOptimizer()
    cv_opt = optimizer.train(X_opt, y_opt, feature_names)
    optimizer.save("outputs/models/field_optimizer.joblib")
    preproc_opt.transformer
    import joblib
    joblib.dump(preproc_opt.transformer, "outputs/models/preprocessor_opt.joblib")
    print(f"  CV R2: {cv_opt:.4f}")
    print(f"  Top features: {dict(list(optimizer.feature_importances().items())[:5])}")

    # --- Allocation Model (production_efficiency) ---
    print("\nTraining Allocation Model (RandomForest) ...")
    preproc_alloc = FieldPreprocessor()
    X_alloc, y_alloc = preproc_alloc.fit_transform(df, ALLOC_TARGET)

    allocator = AllocationModel()
    cv_alloc = allocator.train(X_alloc, y_alloc, feature_names)
    allocator.save("outputs/models/allocation_model.joblib")
    joblib.dump(preproc_alloc.transformer, "outputs/models/preprocessor_alloc.joblib")
    print(f"  CV R2: {cv_alloc:.4f}")
    print(f"  Top features: {dict(list(allocator.feature_importances().items())[:5])}")

    results = {
        "optimizer_cv_r2": round(cv_opt, 4),
        "allocator_cv_r2": round(cv_alloc, 4),
        "dataset_rows": len(df),
        "optimizer_features": len(feature_names),
    }
    with open("outputs/models/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete. Artifacts saved to outputs/models/")
    return results


if __name__ == "__main__":
    main()

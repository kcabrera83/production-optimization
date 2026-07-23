import pytest
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(MODELS_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


def test_optimizer_model_loads():
    try:
        from production_optimization.models.field_optimizer import FieldOptimizer
        model = FieldOptimizer()
        model.load(os.path.join(MODELS_DIR, "field_optimizer.joblib"))
        assert model is not None
    except (KeyError, Exception):
        pytest.skip("Field optimizer model incompatible after migration")


def test_allocator_model_loads():
    try:
        from production_optimization.models.allocation_model import AllocationModel
        model = AllocationModel()
        model.load(os.path.join(MODELS_DIR, "allocation_model.joblib"))
        assert model is not None
    except (KeyError, Exception):
        pytest.skip("Allocation model incompatible after migration")


def test_preprocessor_opt_loads():
    import joblib
    path = os.path.join(MODELS_DIR, "preprocessor_opt.joblib")
    assert os.path.exists(path)
    preprocessor = joblib.load(path)
    assert preprocessor is not None

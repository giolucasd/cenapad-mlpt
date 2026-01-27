from cenapad_mlpt.models.cnn import SimpleCNN

MODEL_REGISTRY = {
    "simple_cnn": SimpleCNN,
}


def build_model_from_config(config: dict):
    name = config["global"]["model_name"]
    model_cfg = config["model"]

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[name](model_cfg)

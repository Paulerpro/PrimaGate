from primagate.client import PrimaGate, acompletion, completion

__version__ = "0.1.0"

# Read on every call by the module-level functions; an instance has its own `drop_params`.
drop_params: bool | list[str] | None = None

__all__ = ["PrimaGate", "__version__", "acompletion", "completion"]

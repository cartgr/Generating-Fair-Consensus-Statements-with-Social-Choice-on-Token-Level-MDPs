from .base import BaseGenerator  # Import base class for type hinting if needed
from .mcts import MCTSGenerator
from .beam_search import BeamSearchGenerator
from .finite_lookahead import FiniteLookaheadGenerator
from .best_of_n import BestOfNGenerator
from .zero_shot import ZeroShotGenerator
from .habermas_machine import HabermasMachineGenerator
from .predefined_statement import PredefinedStatementGenerator

# Map method names (from config) to their corresponding generator classes
GENERATOR_MAP = {
    "mcts": MCTSGenerator,
    "beam_search": BeamSearchGenerator,
    "finite_lookahead": FiniteLookaheadGenerator,
    "best_of_n": BestOfNGenerator,
    "zero_shot": ZeroShotGenerator,
    "habermas_machine": HabermasMachineGenerator,
    "predefined": PredefinedStatementGenerator,
}


def get_method_generator(
    method_name: str, method_config: dict, generation_model: str
) -> BaseGenerator:
    """
    Factory function to get a statement generator instance based on the method name.

    Args:
        method_name: The name of the method (e.g., 'mcts', 'beam_search', 'predefined').
        method_config: A dictionary containing configuration specific to this method.
        generation_model: The identifier for the generation model to be used.

    Returns:
        An instance of a class derived from BaseGenerator.

    Raises:
        ValueError: If the method_name is not found in GENERATOR_MAP.
    """
    generator_class = GENERATOR_MAP.get(method_name)
    if generator_class:
        # Pass the model identifier FIRST, then the specific config for this method
        return generator_class(generation_model, method_config)
    else:
        raise ValueError(f"Unknown method: {method_name}")


# Optional: Define what gets imported when using 'from src.methods import *'
# __all__ = [
#     "get_method_generator",
#     "BaseGenerator",
#     "MCTSGenerator",
#     "BeamSearchGenerator",
#     "FiniteLookaheadGenerator",
#     "BestOfNGenerator",
#     "ZeroShotGenerator",
#     "HabermasMachineGenerator",
#     "PredefinedStatementGenerator",
# ]

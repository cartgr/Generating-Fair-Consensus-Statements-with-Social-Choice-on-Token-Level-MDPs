import logging
from .base import BaseGenerator

logger = logging.getLogger(__name__)


class PredefinedStatementGenerator(BaseGenerator):
    """
    A generator that doesn't generate anything, but instead returns a
    predefined statement provided in its configuration.

    This is useful for evaluating externally generated or reference statements
    within the same experimental framework.
    """

    def __init__(self, model_identifier: str, config: dict):
        """
        Initializes the generator.

        Args:
            model_identifier: The identifier for the generation model (ignored by this generator).
            config: A dictionary containing configuration specific to this method.
                    Expected key: 'predefined_statement' (str).
        """
        # Pass model_identifier for consistency, though it's not used for generation here.
        super().__init__(model_identifier, config)
        self.predefined_statement = self.config.get("predefined_statement")

        if self.predefined_statement is None:
            logger.error(
                f"Configuration for {self.__class__.__name__} is missing the required 'predefined_statement' key."
            )
            # Optionally raise an error or set a default error statement
            # raise ValueError("Missing 'predefined_statement' in config for PredefinedStatementGenerator")
            self.predefined_statement = (
                "[ERROR: Predefined statement not found in config]"
            )

    def generate_statement(self, issue: str, agent_opinions: dict) -> str:
        """
        Returns the predefined statement stored during initialization.

        Args:
            issue: The central issue or topic (ignored).
            agent_opinions: A dictionary mapping agent identifiers to their opinions (ignored).

        Returns:
            The predefined statement provided in the configuration.
        """
        logger.info(
            f"Using predefined statement for '{issue}' via {self.__class__.__name__}"
        )
        logger.debug(f"  Predefined statement: '{self.predefined_statement}'")
        # Simply return the statement loaded from the config
        return self.predefined_statement

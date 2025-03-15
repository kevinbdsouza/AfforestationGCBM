"""
ExpConfig class, which holds configuration parameters
for a simulation experiment.
"""

from typing import Any, Optional


class ExpConfig:
    """
    Configuration for a simulation experiment.

    Attributes:
        num_tiles (int): The number of tiles used in the simulation.
        afforestation_area (Optional[Any]): The area designated for afforestation. Defaults to None.
        fire_params (Optional[Any]): Parameters for fire simulations. Defaults to None.
        yield_perc (str): Yield percentage identifier. Defaults to an empty string.
    """

    def __init__(self) -> None:
        """
        Initialize the experiment configuration with default values.
        """
        self.num_tiles: int = 0
        self.afforestation_area: Optional[Any] = None
        self.fire_params: Optional[Any] = None
        self.yield_perc: str = ""

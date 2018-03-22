from serpent.game import Game

from .api.api import KingdomAPI

from serpent.utilities import Singleton


class SerpentKingdomGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "steam"

        kwargs["window_name"] = "Kingdom"

        kwargs["app_id"] = "368230"
        kwargs["app_args"] = None
        

        super().__init__(**kwargs)

        self.api_class = KingdomAPI
        self.api_instance = None



    @property
    def screen_regions(self):
        regions = {
            "WALLET": (21, 796, 141, 873)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets

import serpent
from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey
from plugins.SerpentKingdomGameAgentPlugin.files.ml_models.KerasModel import KerasDeepKingdom


class SerpentKingdomGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.model = KerasDeepKingdom((360, 627, 3))

    def setup_play(self):
        pass

    def handle_play(self, game_frame):
        print(game_frame.eighth_resolution_frame.shape, game_frame.frame.shape)
        for i, game_frame in enumerate(self.game_frame_buffer.frames):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

        move = self.model.decide(game_frame.frame)
        score = self.model.evaluate_move(move)
        self.model.update_weights(game_frame.frame, score)

        self.input_controller.press_key(move)

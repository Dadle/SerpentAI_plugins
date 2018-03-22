import serpent
from serpent.game_agent import GameAgent
from plugins.SerpentKingdomGameAgentPlugin.files.ml_models.KerasModel import KerasDeepKingdom
from serpent.input_controller import KeyboardKey
from serpent.frame_grabber import FrameGrabber
import numpy as np
import os
import time


class SerpentKingdomGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play
        self.memory_timeframe = 4

    def setup_play(self):
        self.game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        print("game_frame_buffer:", self.game_frame_buffer)
        #self.ppo_agent.generate_action(game_frame_buffer)

        self.window_dim = (self.game.window_geometry['height'], self.game.window_geometry['width'], 3)
        self.model = KerasDeepKingdom(time_dim=(self.memory_timeframe,),
                                      game_frame_dim=self.window_dim)  # (600, 960, 3))#(360, 627, 3))
        print("Screen_regions:", self.game.screen_regions)
        for region in self.game.screen_regions:
            print("Region is", region)
            # TODO: Fix this absolut pointer
            path = "C:\\SerpentAI\datasets\collect_frames"
            classes = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
                          and name.__contains__(region + "_")]
            print("directory contains: ", classes)

            if os.path.isfile(region + "_trained_model.h5"):
                print("Loading model", region, "from file")
                self.game.api_class.SpriteLocator.load_model(model_name=region, classes=classes)
            else:
                print("Building and training", region, "network from scratch")
                self.game.api_class.SpriteLocator.construct_sprite_locator_network(model_name=region,
                                                                                   screen_region=
                                                                                   self.game.screen_regions[region],
                                                                                   classes=classes)
                self.game.api_class.SpriteLocator.train_model(classes=classes, model_name=region)

    def handle_play(self, game_frame):
        self.game_frame_buffer = FrameGrabber.get_frames([0, 1, 2, 3], frame_type="PIPELINE")
        frame_buffer = self.game_frame_buffer.frames

        for model_name in self.game.api_class.SpriteLocator.sprite_models:
            print("model is", model_name)

            print(self.game.api_class.
                  SpriteLocator.sprite_recognized(game_frame=game_frame,
                                                  screen_region=self.game.screen_regions[model_name],
                                                  model_name=model_name,  #screen_region_frame,
                                                  classes=self.game.api_class.SpriteLocator.
                                                  sprite_models[model_name]["classes"]))

        print(game_frame.frame.shape)
        for i, game_frame in enumerate(frame_buffer):
            self.visual_debugger.store_image_data(
                game_frame.frame,
                game_frame.frame.shape,
                str(i)
            )

        #self.game_frame_buffer.append(game_frame.frame)
        #print("game_frame_buffer:", frame_buffer)
        #if len(frame_buffer) >= self.memory_timeframe:
            # there is enough frames stored to train the network

        move_per_timestep = self.model.decide(frame_buffer)
        score = self.model.evaluate_move(move_per_timestep)
        self.model.update_weights(frame_buffer, score)
       # else:
            # this is a new game, so just start going right
            #move = KeyboardKey.KEY_RIGHT

        #self.input_controller.press_key(move_per_timestep[len(move_per_timestep)-1])

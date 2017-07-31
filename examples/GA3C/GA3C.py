# --------------------------------------------------------
# GA3C for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from Config import Config
from Server import Server

# Adjust configs for Play mode
if Config.PLAY_MODE:
    Config.AGENTS = 1
    Config.PREDICTORS = 1
    Config.TRAINERS = 1
    Config.DYNAMIC_SETTINGS = False

    Config.LOAD_CHECKPOINT = True
    Config.TRAIN_MODELS = False
    Config.SAVE_MODELS = False

if __name__ == "__main__":

    import dragon.config
    dragon.config.EnableCUDA()

    Server().main()

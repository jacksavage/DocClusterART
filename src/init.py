import json
import logging as log

def run():
    # load the configuration file
    with open("config.json") as file:
        config = json.load(file)

    # configure the logger
    log.basicConfig(
        level=config["log_level"],
        format="[%(asctime)s] %(message)s",
        datefmt="%I:%M:%S %p"
    )
    log.info("App started")

    # pass the configuration to the app
    return config

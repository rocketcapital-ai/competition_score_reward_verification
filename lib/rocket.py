from pathlib import Path

from lib.blockchain import Competition

COMPETITION_NAME = 'ROCKET'
COMPETITION_GENESIS_BLOCK = 19664491
DATA_DIR = Path("data")


def get_rocket_competition() -> Competition:
    return Competition(COMPETITION_NAME, COMPETITION_GENESIS_BLOCK, DATA_DIR)

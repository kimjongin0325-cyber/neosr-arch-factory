
from neosr.utils.rng import rng

# patched: prevent argparse from running on import
def get_rng():
    try:
        return rng()
    except:
        return None

rng = None

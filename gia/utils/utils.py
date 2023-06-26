from transformers.utils import logging
from uncertainties import ufloat
from uncertainties.core import Variable


def ufloat_encoder(obj):
    if isinstance(obj, Variable):
        return {"__ufloat__": True, "n": obj.n, "s": obj.s}
    return obj


def ufloat_decoder(dct):
    if "__ufloat__" in dct:
        return ufloat(dct["n"], dct["s"])
    return dct


logger = logging.get_logger("gia")
logger.setLevel(logging.INFO)


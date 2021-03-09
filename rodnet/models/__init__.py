from .rodnet_c21d import RODNetC21D
from .rodnet_cdc import RODNetCDC
from .rodnet_cdc_deep import RODNetCDC as RODNetCDC_deep
from .rodnet_gsc import GSCStack
from .rodnet_hg import RODNetHG
from .rodnet_hgwi import RODNetHGwI


def get_model(name):
    if name == "CDC":
        return RODNetCDC
    elif name == "HG":
        return RODNetHG
    elif name == "HGwI":
        return RODNetHGwI
    elif name == "C21D":
        return RODNetC21D
    elif name == "CDCD":
        return RODNetCDC_deep
    elif name == "GSC":
        return GSCStack
    else:
        raise NotImplementedError

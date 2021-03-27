from .resnet import Resnet18
from .rodnet_c21d import RODNetC21D
from .rodnet_cdc import RODNetCDC
from .rodnet_cdc_deep import RODNetCDC as RODNetCDC_deep
from .rodnet_gsc import RODNetGSC
from .rodnet_gscmp import RODNetGSCmp
from .rodnet_hg import RODNetHG
from .rodnet_hgwi import RODNetHGwI
from .rodnet_resnet import RODNetResnet18, RODNetResnet18_b, RODNetResnet18_c


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
        return RODNetGSC
    elif name == "GSCmp":
        return RODNetGSCmp
    elif name == "cls_resnet":
        return Resnet18
    elif name == "Resnet18":
        return RODNetResnet18
    elif name == "Resnet18b":
        return RODNetResnet18_b
    elif name == "Resnet18c":
        return RODNetResnet18_c
    else:
        raise NotImplementedError

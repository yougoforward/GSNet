from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .psp_att import *
from .encnet import *
from .deeplabv3 import *
from .deeplabv3_att import *
from .deeplabv3plus import*
from .gsnet import *
from .gsnet_noatt import *
from .gsnet_noduide import *
from .psp_att_noguide import *
from .deeplabv3_att_noguide import *
from .gsnet_noatt_nose import *
from .gsnet_noatt_nose_nopsaa import *
from .gsnet2 import *
from .gsnet3 import *
from .gsnet4 import *
from .new_psp3 import *
from .new_psp3_noguide import *
from .new_psp3_noatt import *
from .new_psp3_noatt_nose import *
from .new_psp3_noatt_nose_nopsaa import *
from .new_psp3_nose import *
from .new_psp3_aspp_base import *
from .new_psp3_aspp_base_att import *
from .new_psp3_aspp_base_psaa import *
from .new_psp3_aspp_base_psaa_att import *
from .new_psp3_nopsaa import *
from .new_psp3_nose_nopsaa import *
from .new_psp3_noatt_nopsaa import *
from .fcn_att import *
from .gsnet_nose import *
from .new_psp3_nopsaa_nose_nogp import *
from .gsnet6 import *
from .gsnet7 import *
from .gsnet7_nose import *
from .gsnet8 import *
from .mlgsnet import *
def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'fcn_att': get_fcn_att,
        'encnet': get_encnet,
        'deeplabv3plus': get_deeplabv3plus,
        'deeplabv3': get_deeplabv3,
        'deeplabv3_att': get_deeplabv3_att,
        'psp': get_psp,
        'psp_att': get_psp_att,
        'gsnet': get_gsnet,
        'gsnet_noatt': get_gsnet_noatt,
        'gsnet_noguide': get_gsnet_noguide,
        'psp_att_noguide': get_psp_att_noguide,
        'deeplabv3_att_noguide': get_deeplabv3_att_noguide,
        'gsnet_noatt_nose': get_gsnet_noatt_nose,
        'gsnet_noatt_nose_nopsaa':get_gsnet_noatt_nose_nopsaa,
        'gsnet2': get_gsnet2,
        'gsnet3': get_gsnet3net,
        'gsnet4': get_gsnet4net,
        'new_psp3': get_new_psp3net,
        'new_psp3_noguide': get_new_psp3_noguidenet,
        'new_psp3_noatt': get_new_psp3_noattnet,
        'new_psp3_noatt_nose': get_new_psp3_noatt_nosenet,
        'new_psp3_noatt_nose_nopsaa': get_new_psp3_noatt_nose_nopsaanet,
        'new_psp3_nose': get_new_psp3_nosenet,
        'new_psp3_aspp_base': get_new_psp3_aspp_basenet,
        'new_psp3_aspp_base_att': get_new_psp3_aspp_base_attnet,
        'new_psp3_aspp_base_psaa': get_new_psp3_aspp_base_psaanet,
        'new_psp3_aspp_base_psaa_att': get_new_psp3_aspp_base_psaa_attnet,
        'new_psp3_nopsaa': get_new_psp3_nopsaanet,
        'new_psp3_nose_nopsaa': get_new_psp3_nose_nopsaanet,
        'new_psp3_noatt_nopsaa': get_new_psp3_noatt_nopsaanet,
        'gsnet_nose': get_gsnet_nose,
        'new_psp3_nopsaa_nose_nogp': get_new_psp3_nopsaa_nose_nogpnet,
        'gsnet6': get_gsnet6net,
        'gsnet7': get_gsnet7net,
        'gsnet7_nose': get_gsnet7_nosenet,
        'gsnet8': get_gsnet8net,
        'mlgsnet': get_mlgsnetnet,
    }
    return models[name.lower()](**kwargs)

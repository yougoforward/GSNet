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
from .gsnet_no_att_nose_nopsaa import *
from .gsnet2 import *
from .gsnet3 import *
from .new_psp3 import *
from .new_psp3_noguide import *
from .new_psp3_noatt import *
from .new_psp3_noatt_nose import *
from .new_psp3_noatt_nose_nopsaa import *
from .

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
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
        'new_psp3': get_new_psp3net,
        'new_psp3_noguide': get_new_psp3_noguidenet,
        'new_psp3_noatt': get_new_psp3_noattnet,
        'new_psp3_noatt_nose': get_new_psp3_noatt_nosenet,
        'new_psp3_noatt_nose_nopsaa': get_new_psp3_noatt_nose_nopsaanet,

    }
    return models[name.lower()](**kwargs)

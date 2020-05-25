from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .psp import *
from .encnet import *
from .deeplabv3 import *
from .deeplabv3plus import*
from .can import *
from .can2 import *
from .can3 import *
from .can4 import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'deeplab': get_deeplab,
        'deeplabv3plus': get_deeplabv3plus,
        'can': get_can,
        'can2': get_can2,
        'can3': get_can3,
        'can4': get_can4,
    }
    return models[name.lower()](**kwargs)

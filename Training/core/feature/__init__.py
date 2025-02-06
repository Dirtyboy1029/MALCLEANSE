from .feature_extraction import DrebinFeature

from collections import namedtuple
from ..model_lib import model_name_type_dict

feature_type_scope_dict = {
    'drebin': DrebinFeature,
}

# bridge the gap between the feature extraction and dnn architecture
_ARCH_TYPE = namedtuple('architectures', model_name_type_dict.keys())
_architecture_feature_extraction = _ARCH_TYPE(dnn='drebin')
_architecture_feature_extraction_dict = dict(_architecture_feature_extraction._asdict())
feature_type_vs_architecture = dict(zip(_architecture_feature_extraction_dict.values(),
                                        _architecture_feature_extraction_dict.keys()))

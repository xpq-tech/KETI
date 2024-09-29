from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.unke import UnkeHyperParams, apply_unke_to_model
from ..models.non_edit import NON_EDITHyperParams, apply_non_edit_to_model

ALG_DICT = {
    "FT": apply_ft_to_model,
    "GRACE": apply_grace_to_model,
    "UNKE": apply_unke_to_model,
    "non-edit": apply_non_edit_to_model
}
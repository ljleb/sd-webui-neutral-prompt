from typing import List, Optional
from lib_neutral_prompt import neutral_prompt_parser


is_enabled: bool = False
prompt_exprs: List[neutral_prompt_parser.PromptExpr] = []
cfg_rescale: float = 0.0
verbose: bool = True
cfg_rescale_override: Optional[float] = None


def apply_and_clear_cfg_rescale_override():
    global cfg_rescale, cfg_rescale_override
    if cfg_rescale_override is not None:
        cfg_rescale = cfg_rescale_override
        cfg_rescale_override = None

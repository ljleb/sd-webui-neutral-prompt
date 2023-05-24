from typing import List
from lib_neutral_prompt import perp_parser


is_enabled: bool = False
prompt_exprs: List[perp_parser.Prompt] = []
cfg_rescale: float = 0.
verbose: bool = True

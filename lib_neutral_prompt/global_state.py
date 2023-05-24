from typing import List
from lib_neutral_prompt import neutral_prompt_parser


is_enabled: bool = False
prompt_exprs: List[neutral_prompt_parser.NeutralPrompt] = []
cfg_rescale: float = 0.
verbose: bool = True

import sys
# Silence Python warnings (FutureWarning, DeprecationWarning, etc.) unless --verbose.
# Must run before any ML library imports since most warnings fire at import time.
# We keep Gradio chatter, HF/torch progress bars, and generation tqdm intact.
if '--verbose' not in sys.argv:
    import os as _os
    _os.environ.setdefault('PYTHONWARNINGS', 'ignore')
    import warnings as _warnings
    _warnings.filterwarnings('ignore')

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.interface.gradio import create_ui
from stable_audio_tools.verbose import set_verbose
import json

import torch

def main(args):
    set_verbose(getattr(args, 'verbose', False))
    torch.manual_seed(42)

    interface = create_ui(
        model_config_path = args.model_config,
        ckpt_path=args.ckpt_path,
        pretrained_name=args.pretrained_name,
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        gradio_title=args.title,
        lora_ckpt_paths=args.lora_ckpt_path,
        default_prompt=args.default_prompt,
    )
    interface.queue()
    interface.launch(share=args.share, auth=(args.username, args.password) if args.username is not None else None, js=getattr(interface, '_sao_js', None), theme=getattr(interface, '_sao_theme', None))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False, default=True)
    parser.add_argument('--title', type=str, help='Display Title top of Gradio', required=False)
    parser.add_argument('--lora-ckpt-path', type=str, nargs='*', help='Path(s) for LoRA(s) to apply. Can specify multiple.', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--default-prompt', type=str, default=None,
                        help='Default prompt to pre-fill in the textbox')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print detailed load/generation progress')
    args = parser.parse_args()
    main(args)
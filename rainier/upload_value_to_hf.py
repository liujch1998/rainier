import torch
from args import get_args
from model.value import Value
import os

args = get_args()
value = Value(
    model_type=args.model_type,
    model_ckpt=None,
    device='cpu',
    device_map=None,
)
checkpoint = torch.load(args.load_from_ckpt, map_location=torch.device('cpu'))
value.model.load_state_dict(checkpoint['value_model'])

value.model.push_to_hub('liujch1998/rainier-large-value', use_auth_token=os.environ.get('HF_TOKEN_UPLOAD'))


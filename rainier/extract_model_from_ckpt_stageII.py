import torch
from args import get_args
from model.policy import Policy

args = get_args()
policy = Policy(
    model_type=args.model_type,
    model_ckpt=args.model_ckpt,
    policy_value_sharing=args.policy_value_sharing,
    max_input_len=args.max_input_len,
    max_output_len=args.max_output_len,
    device='cpu',
)
checkpoint = torch.load(args.load_from_ckpt, map_location=torch.device('cpu'))
policy.model.load_state_dict(checkpoint['policy_model'])

torch.save(policy.model.state_dict(), f'../model/rainier-large.pth')


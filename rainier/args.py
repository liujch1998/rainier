import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument(
        '--mode', type=str, choices=['train', 'eval'], required=True, help='train or eval?')

    # dataset
    parser.add_argument(
        '--train_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg')
    parser.add_argument(
        '--eval_tasks', type=str, default='obqa,arc_e,arc_h,ai2sci_e,ai2sci_m,csqa,qasc,piqa,siqa,wg,numersense,riddlesense,quartz,hellaswag')
    parser.add_argument(
        '--eval_split', type=str, default='dev', choices=['dev', 'test'])

    # model
    parser.add_argument(
        '--model_type', type=str, default='t5-large', help='model used for policy, ref policy, and value')
    parser.add_argument(
        '--model_ckpt', type=str, default='../model/rainier-large_stageI.pth', help='model ckpt used for policy and ref policy (NOT value!)')
    parser.add_argument(
        '--use_model_ckpt_for_value', action='store_true', default=False)
    parser.add_argument(
        '--policy_value_sharing', action='store_true', default=False)
    parser.add_argument(
        '--qa_model_type', type=str, default='allenai/unifiedqa-t5-large', help='model used for QA')
    parser.add_argument(
        '--qa_model_ckpt', type=str, default=None, help='model ckpt used for QA')
    parser.add_argument(
        '--max_input_len', type=int, default=256, help='max length of the input prompt')
    parser.add_argument(
        '--max_output_len', type=int, default=32, help='max length of the output knowledge')
    parser.add_argument(
        '--load_from_ckpt', type=str, default=None, help='ckpt path to resume training or run eval')
    parser.add_argument(
        '--eval_ckpt', type=str, default='../model/rainier-large.pth', help='rainier ckpt to run eval')

    # reward
    parser.add_argument(
        '--kl_coef', type=float, default=0.2, help='coefficient for KL term in reward')
    parser.add_argument(
        '--reward_shape', type=int, default=4, help='refer to reward.py for implementation of each option')
    parser.add_argument(
        '--gain', type=float, default=None, help='precomputed normalization factor for reward')
    parser.add_argument(
        '--bias', type=float, default=None, help='precomputed normalization factor for reward')

    # ppo
    parser.add_argument(
        '--pg_coef', type=float, default=1.0, help='policy loss coefficient')
    parser.add_argument(
        '--vf_coef', type=float, default=1.0, help='value loss coefficient')
    parser.add_argument(
        '--cliprange', type=float, default=.2, help='clip parameter for policy gradient')
    parser.add_argument(
        '--cliprange_value', type=float, default=.2, help='clip parameter for value function')
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='discount factor for rewards')
    parser.add_argument(
        '--lam', type=float, default=0.95, help='lambda parameter for generalized advantage estimation')
    parser.add_argument(
        '--whiten_rewards', action='store_false', default=True, help='whether to normalize reward in each minibatch')
    parser.add_argument(
        '--clip_grad', action='store_true', default=False, help='whether to clip gradient')
    parser.add_argument(
        '--max-grad-norm', type=float, default=0.5, help='maximum norm of gradients ')

    # train
    parser.add_argument(
        '--total_episodes', type=int, default=1000000, help='total number of episodes')
    parser.add_argument(
        '--num_warmup_step_ratio', type=float, default=0.0, help = 'ratio of number of steps to use for warmup with linear warmup')
    parser.add_argument(
        '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument(
        '--noptepochs', type=int, default=4, help='number of ppo epochs reusing rollouts')
    parser.add_argument(
        '--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument(
        '--temperature', type=float, default=0.7, help='temperature for sampling from policy during training')

    # eval
    parser.add_argument(
        '--num_samples', type=int, default=10, help='number of knowledges to sample during eval')
    parser.add_argument(
        '--top_p', type=float, default=0.5, help='hyperparameter for nucleus sampling')
    parser.add_argument(
        '--ensembling', type=str, default='max', choices=['max', 'moe', 'poe'], help='ensembling method for inference')

    # other
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log_interval', type=int, default=100, help='step interval to print out logs')
    parser.add_argument(
        '--save_interval', type=int, default=500, help='step interval to save model checkpoints')
    parser.add_argument(
        '--eval_interval', type=int, default=500, help='step interval to do evaluation')
    parser.add_argument(
        '--nosave', default=False, action='store_true')
    parser.add_argument(
        '--eval_baseline', action='store_true', help='whether to evaluate the no-knowledge baseline')
    parser.add_argument(
        '--cuda_deterministic', action='store_false', default=True,
        help='sets flags for determinism when using CUDA (potentially slow!)')

    args = parser.parse_args()

    return args


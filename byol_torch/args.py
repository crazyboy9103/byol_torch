import argparse

def tobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_tau",    type=float,  default=4e-3)
    parser.add_argument('--base_lr',     type=float,  default=1e-3)
    parser.add_argument('--epochs',      type=int,    default=100)
    parser.add_argument('--batch_size',  type=int,    default=1024)

    # mlp
    parser.add_argument("--output_dim",  type=int,    default=512)
    parser.add_argument("--proj_dim",    type=int,    default=256)
    parser.add_argument("--pred_dim",    type=int,    default=256)
    parser.add_argument("--hidden_dim",  type=int,    default=512)

    # dataset
    parser.add_argument('--dataset',     type=str,    default='CIFAR10',  choices=['CIFAR10','CIFAR100'])

    # backbone
    parser.add_argument('--model',       type=str,    default='resnet18',  choices=['resnet18','resnet50'])
    parser.add_argument('--pretrained',  type=tobool, default=False)
    parser.add_argument('--ckpt_path',   type=str,    default="./checkpoints/checkpoint.pth.tar")
    
    parser.add_argument('--wandb_name',   type=str,   default="byol_implement")
    parser.add_argument('--type',         type=str,   default="pretrain",  choices=["pretrain", "linear", "semi"])
    parser.add_argument('--linear_lr',    type=float, default=1e-3)
    parser.add_argument('--linear_epoch', type=int,   default=5)

    
    return parser.parse_args(args=[]) 

import argparse

def get_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # imbalanced related
    # LDS
    parser.add_argument('--lds', action='store_true', default=False, help='whether to enable LDS')
    parser.add_argument('--d', type=str, default='')
    parser.add_argument('--lds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='LDS kernel type')
    parser.add_argument('--lds_ks', type=int, default=9, help='LDS kernel size: should be odd number')
    parser.add_argument('--lds_sigma', type=float, default=1, help='LDS gaussian/laplace kernel sigma')
    # FDS
    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=9, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=3, choices=[0, 3],
                        help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

    # re-weighting: SQRT_INV / INV
    parser.add_argument('--reweight', type=str, default='none', choices=['none', 'sqrt_inv', 'inverse'], help='cost-sensitive reweighting scheme')
    # two-stage training: RRT
    parser.add_argument('--retrain_fc', action='store_true', default=False, help='whether to retrain last regression layer (regressor)')

    # training/optimization related
    parser.add_argument('--dataset', type=str, default='agedb', choices=['imdb_wiki', 'agedb'], help='dataset name')
    parser.add_argument('--data_dir', type=str, default='./data', help='data directory')
    parser.add_argument('--model', type=str, default='resnet50', help='model name')
    parser.add_argument('--store_root', type=str, default='checkpoint/', help='root path for storing checkpoints, logs')
    parser.add_argument('--store_name', type=str, default='', help='experiment store name')
    parser.add_argument('--exp', type=str, default='Resnet_1', help='experiment store name')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'], help='optimizer type')
    parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1', 'focal_l1', 'focal_mse', 'huber', 'differential','crossent'], help='training loss type')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epoch', type=int, default=80, help='number of epochs to train')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='optimizer weight decay')
    parser.add_argument('--schedule', type=int, nargs='*', default=[60, 80], help='lr schedule (when to drop lr by 10x)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--print_freq', type=int, default=10, help='logging frequency')
    parser.add_argument('--img_size', type=int, default=224, help='image size used in training')
    parser.add_argument('--workers', type=int, default=32, help='number of workers used in data loading')
    # checkpoints
    parser.add_argument('--resume', type=str, default='', help='checkpoint file path to resume training')
    parser.add_argument('--pretrained', type=str, default='', help='checkpoint file path to load backbone weights')
    parser.add_argument('--evaluate', action='store_true', help='evaluate only flag')
    parser.add_argument('--backbone', type=str, default='vgg16', help='evaluate only flag')
    parser.add_argument('--tem', type=int, default=0, help='evaluate only flag')
    parser.set_defaults(augment=True)

    return parser
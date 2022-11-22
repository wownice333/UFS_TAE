import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='UFS-TAE Arguments.')
    parser.add_argument('--datasetname', dest='DS', default='Yale_32x32', help='Dataset')
    parser.add_argument('--path', dest='path', default='./dataset', help='Data Path')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.01,
                        help='Coefficient of L2,1-norm')
    parser.add_argument('--beta', dest='beta', type=float, default=0.01,
                        help='Coefficient of Orthogonality Constraint')
    parser.add_argument('--k', dest='k', type=int, default=5)
    parser.add_argument('--repNum', dest='repNum', type=int,
                        help='Repeat number.', default=20)
    parser.add_argument('--epochs', dest='epochs', type=int, help='Training Epochs', default=500)
    parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[400, 450], help='Learning Rate Milestones')

    return parser.parse_args()

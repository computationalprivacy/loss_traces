import argparse
import time
import random
import numpy as np
import torch
from opacus.grad_sample import GradSampleModule

from data_processing.data_processing import prepare_transform, get_trainset, get_testset, prepare_loaders, \
    get_num_classes
from models.model import load_model
from trainer import Trainer


def parse_input():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batchsize', default=64, type=int, help='batch size')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='base learning rate')
    parser.add_argument('--exp_id', default='exp_default', type=str, help='filename for saving results')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0, type=float, help='value of momentum')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--arch', default='simple_convnet', type=str, help='model architecture to use')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset to be trained on')
    parser.add_argument('--clip_norm', type=float, default=None,help='enable per-sample gradient clipping and set clipping norm')
    parser.add_argument('--private', action='store_true')
    parser.add_argument('--augment', action='store_true', help='train with augmentation if available')
    parser.add_argument('--checkpoint', action='store_true')

    # For target model training
    parser.add_argument('--track_free_loss', action='store_true',
                        help='track individual losses from training (train & test sets)')
    parser.add_argument('--track_grad_norms', action='store_true',
                        help='enable individual average grad norm tracking every epoch')
    parser.add_argument('--start_tracking', '-st', default=0, type=int, help='epoch to start individual norm tracking')
    parser.add_argument('--track_for', '-tf', default=1, type=int,
                        help='number of epochs to track individual norms for')

    parser.add_argument('--balanced_sampling', action='store_false',
                        help='force target model to train on *balanced* subset of training data')

    # For training a model on same set as the target
    parser.add_argument('--dual', default=None, type=str,
                        help='To train a model on an existing set of indices (and name this model)')

    # For shadow model training
    parser.add_argument('--shadow_count', default=None, type=int, help='number of shadow models being trained')
    parser.add_argument('--shadow_id', default=None, type=int, help='id of the individual shadow model to train')
    parser.add_argument('--gpu', default='', type=str, help="select gpu to run e.g. ':0' where multiple are available")

    parser.add_argument('--model_start', type=int, default=0)
    parser.add_argument('--model_stop', type=int, default=1)

    args = parser.parse_args()

    if (args.shadow_id is not None and not args.shadow_count) \
            or (args.dual and args.shadow_count is not None):
        raise ValueError('Invalid argument combination')

    return args


def set_seed(seed=0):
    # set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    args = parse_input()

    device = "cuda" + args.gpu if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    train_transform = prepare_transform(args.dataset, args.arch, args.augment)
    plain_transform = prepare_transform(args.dataset, args.arch)

    start = time.time()
    train_superset = get_trainset(args.dataset, train_transform)
    plain_train_superset = get_trainset(args.dataset,plain_transform)
    testset = get_testset(args.dataset, plain_transform)
    stop = time.time()
    print(f'Loading datasets took {stop - start}')

    for model_id in range(args.model_start, args.model_stop):
        print(f'-------- Training model {model_id} --------')
        if args.shadow_count:
            args.shadow_id = model_id

        set_seed(args.seed)
        print('==> Preparing data..')

        num_classes = get_num_classes(args.dataset)
        trainloader, plainloader, testloader = prepare_loaders(train_superset, plain_train_superset, testset,
                                                               num_classes, args)

        num_training = len(trainloader.dataset)
        num_test = len(testloader.dataset)
        print('Training on: ', num_training, 'Testing on: ', num_test)

        steps_per_epoch = (num_training // args.batchsize)
        if (num_training % args.batchsize != 0):
            steps_per_epoch += 1

        print('\n==> Initialising the model..', args.checkpoint)

        model = load_model(args.arch, num_classes).to(device)
        if args.track_grad_norms:
            model = GradSampleModule(model)

        trainer = Trainer(args, (trainloader, plainloader, testloader), device)
        trainer.train_test(model, args, model_id)
    print('\n==> Finished training')


if __name__ == '__main__':
    main()
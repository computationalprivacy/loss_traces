import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer

from config import MODEL_DIR, STORAGE_DIR, LOCAL_DIR


class Trainer:
    def __init__(self, args, dataloaders, device):
        self.args = args
        self.trainloader, self.plainloader, self.testloader = dataloaders
        self.num_epochs = args.epochs
        self.device = device
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')


    @staticmethod
    def get_optimizer(training_params, lr, momentum, weight_decay):
        optimizer = torch.optim.SGD(
            training_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        return optimizer

    @staticmethod
    def get_training_params(model):
        num_params = 0
        num_training_params = 0
        training_params = []
        for p in model.parameters():
            num_params += p.numel()
            if p.requires_grad:
                training_params.append(p)
                num_training_params += p.numel()
        print(f'Training {num_training_params / (10 ** 6)}M parameters out of {num_params / (10 ** 6)}M')
        return training_params

    def get_all_losses_and_grad_norms(self, model, args):
        # model.eval()
        loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')
        model.train()

        all_losses = []
        all_norms = []
        for inputs, targets, _indices in self.plainloader:
            self.optimizer.zero_grad()
            model.zero_grad()
            # if isinstance(self.optimizer, DPOptimizer):
            #     self.optimizer.expected_batch_size = inputs.shape[0]
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = loss_func_sample(outputs, targets)
            all_losses.append(losses)
            losses.mean().backward()
            batch_grads = [p.grad_sample.view(p.grad_sample.size(0), -1) for p in self.training_params]
            batch_norms = torch.cat(batch_grads, dim=1).norm(dim=1)
            all_norms.append(batch_norms)
        all_losses = torch.cat(all_losses, dim=0)
        all_norms = torch.cat(all_norms, dim=0)

        return all_losses.tolist(), all_norms.tolist()


    def train_epoch(self, model, epoch, args):
        print('\nEpoch: %d' % epoch)
        model.train()
        total = 0
        correct = 0
        t0 = time.time()
        free_train_losses = pd.DataFrame(np.full((len(self.trainloader.dataset), 1), np.nan))
        free_train_losses.index = self.trainloader.dataset.indices
        sample_losses, sample_grad_norms = [], []

        # collect init and final grad norms & losses
        if args.track_grad_norms and epoch == 0:
            losses, norms = self.get_all_losses_and_grad_norms(model, args)
            sample_losses.append(losses)
            sample_grad_norms.append(norms)

        # else:
        for batch_idx, (inputs, targets, indices) in enumerate(self.trainloader):
            model.zero_grad()
            self.optimizer.zero_grad()
            # if isinstance(self.optimizer, DPOptimizer):
            #     self.optimizer.expected_batch_size = inputs.shape[0]

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = model(inputs).squeeze(-1).squeeze(-1)

            losses = self.loss_func_sample(outputs, targets)
            if args.track_free_loss:
                free_train_losses.loc[indices.tolist()] = losses.tolist()


            losses.mean().backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            minibatch_correct = predicted.eq(targets.data).float().cpu()
            total += targets.size(0)
            correct += minibatch_correct.sum()

        if args.track_grad_norms:
            losses, norms = self.get_all_losses_and_grad_norms(model, args)
            sample_losses.append(losses)
            sample_grad_norms.append(norms)

        t1 = time.time()
        self.scheduler.step()
        acc = (100. * float(correct) / float(total)) if total > 0 else 0.0
        print('Time: %d s' % (t1 - t0), 'train acc:', acc, end=' ')
        return acc, free_train_losses, sample_losses, sample_grad_norms

    def train_test(self, model, args, model_id):
        self.training_params = self.get_training_params(model)
        self.optimizer = self.get_optimizer(self.training_params, args.lr, args.momentum, args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)

        ## TODO: Define appropriate epsilon and delta values
        if args.private:
            privacy_engine = PrivacyEngine()

            model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(max_grad_norm=10.0, module=model, optimizer=self.optimizer, data_loader=self.trainloader, target_epsilon=8.0, target_delta=1e-5, epochs=args.epochs)

        if args.clip_norm:
            self.optimizer = DPOptimizer(
                optimizer=self.optimizer,
                noise_multiplier=0.0,
                max_grad_norm=args.clip_norm,
                expected_batch_size=args.batchsize,
            )


        # init loss & norm stores
        sample_grad_norms = []
        sample_losses = []
        free_train_losses = pd.DataFrame(np.full((len(self.trainloader.dataset), args.epochs), np.nan))
        free_train_losses.index = self.trainloader.dataset.indices
        free_test_losses = []

        num_training = len(self.trainloader.dataset)
        num_test = len(self.testloader.dataset)
        print('Training on: ', num_training, 'Testing on: ', num_test)

        steps_per_epoch = (num_training // args.batchsize)
        if (num_training % args.batchsize != 0):
            steps_per_epoch += 1

        print(f'-------- Training model {model_id} --------')
        print('\n==> Starting training')

        for epoch in range(args.epochs):
            train_acc, train_fl, train_sl, train_sgn = self.train_epoch(model, epoch, args)
            test_acc, test_fl = self.test(model, args)

            sample_losses.extend(train_sl)
            sample_grad_norms.extend(train_sgn)

            if args.track_free_loss:
                free_train_losses = pd.concat([free_train_losses, train_fl], axis=1)
                free_test_losses.append(torch.cat(test_fl, dim=0).tolist())

            if args.checkpoint and (epoch + 1) % 5 == 0:
                save_model(model, args, self.trainloader, train_acc, test_acc, checkpoint=True, epoch=epoch)


        if args.track_free_loss:
            save_free_loss(free_train_losses, free_test_losses, args)

        if args.track_grad_norms:
            save_tracking_data(sample_losses, sample_grad_norms, args)

        save_model(model, args, self.trainloader, train_acc, test_acc)

    def test(self, model, args):
        loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _indices in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                correct += correct_idx.sum()

        acc = 100. * float(correct) / float(total)
        print('test acc:', acc)

        free_losses = []

        with torch.no_grad():
            for inputs, targets, _indices in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                losses = loss_func_sample(outputs, targets)

                if args.track_free_loss:
                    free_losses.append(losses)

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct_idx = predicted.eq(targets.data).cpu()
                correct += correct_idx.sum()

        acc = 100. * float(correct) / float(total)
        return acc, free_losses


def save_free_loss(train_loss, test_loss, args):
    train_dir = os.path.join(STORAGE_DIR, 'free_train_losses')
    os.makedirs(train_dir, exist_ok=True)
    test_dir = os.path.join(STORAGE_DIR, 'free_test_losses')
    os.makedirs(test_dir, exist_ok=True)
    file = args.exp_id

    if args.dual:
        file += f'_dual_{args.dual}'
    file += '.pq'

    train_path = os.path.join(train_dir, file)
    test_path = os.path.join(test_dir, file)

    train_loss.to_parquet(train_path)
    pd.DataFrame(test_loss).transpose().to_parquet(test_path)


def save_tracking_data(sample_losses, sample_grad_norms, args):
    outdir = os.path.join(LOCAL_DIR, 'grad_norms')
    outdir_loss = os.path.join(STORAGE_DIR, 'losses')
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_loss, exist_ok=True)
    file = args.exp_id
    if args.dual:
        file += f'_dual_{args.dual}'
    elif args.shadow_count:
        file += f'_shadow_{str(args.shadow_id)}'
    file += '.pq'
    fullpath = os.path.join(outdir, file)
    fullpath_loss = os.path.join(outdir_loss, file)
    loss_df = pd.DataFrame(sample_losses).transpose()
    loss_df.columns = loss_df.columns.astype(str)
    loss_df.to_parquet(fullpath_loss)
    # if os.path.exists(fullpath):
    #     print("'DUPLICATE' GRAD NORMS - WILL NOT OVERWRITE PREVIOUS", file=sys.stderr)
    # while os.path.exists(fullpath):
    #     fullpath += "_"
    grad_df = pd.DataFrame(sample_grad_norms).transpose()
    grad_df.columns =  grad_df.columns.astype(str)
    grad_df.to_parquet()


def save_model(model, args, trainloader, train_acc, test_acc, checkpoint=False, epoch=None):
    if args.shadow_count:
        save_name = 'shadow_' + str(args.shadow_id)
    elif args.dual:
        save_name = f'dual_{args.dual}'
    elif args.track_grad_norms:
        save_name = 'target'
    else:
        save_name = 'model'

    model_state_dict = model.state_dict()

    if isinstance(model, GradSampleModule):
        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text

        model_state_dict_wrapped = model_state_dict
        model_state_dict = {}
        for k, v in model_state_dict_wrapped.items():
            model_state_dict[remove_prefix(k, '_module.')] = v

    dic = {
        'model_state_dict': model_state_dict,
        'hyperparameters': vars(args),
        'trained_on_indices': trainloader.dataset.indices,
        'arch': args.arch,
        'seed': args.seed,
        'dataset': args.dataset,
        'train_acc': train_acc,
        'test_acc': test_acc
    }

    dir = os.path.join(MODEL_DIR, args.exp_id)

    if checkpoint:
        print(f"SAVING CHECKPOINT @ EPOCH: {epoch+1}")
        dir = os.path.join(dir, f'checkpoint_before_{epoch + 1}')
    os.makedirs(dir, exist_ok=True)
    fullpath = os.path.join(dir, save_name)
    torch.save(dic, fullpath)

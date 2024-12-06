import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch
from torch.nn import Module
from torch.utils.data import Subset, DataLoader

from config import MODEL_DIR, STORAGE_DIR
from data_processing.data_processing import get_no_shuffle_train_loader, get_num_classes
from models.model import load_model


@dataclass
class AttackConfig:
    exp_id: str
    target_id: str
    checkpoint: Optional[str]
    arch: str
    dataset: str
    augment: bool =  False
    batchsize: int = 500
    num_workers: int = 8
    gpu: str = ''


class ModelConfidenceExtractor:
    def get_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader,  metric="logits")
    def get_scaled_logits(self, model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metric="logits", scale=True)
    def get_losses(self,  model: Module, device: torch.device, loader: DataLoader):
        return self._get_metrics(model, device, loader, metric="losses")

    @staticmethod
    def _get_metrics(model: Module, device: torch.device, loader: DataLoader, scale=False, metric="losses") -> List[float]:
        """Extract metrics from model predictions."""
        model.eval()
        metrics = []

        with torch.no_grad():
            for inputs, targets, _indices in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(-1).squeeze(-1)

                if metric == "losses":
                    loss_func_sample = torch.nn.CrossEntropyLoss(reduction='none')
                    m = loss_func_sample(outputs, targets)

                elif metric == "logits":
                    pred_confs, _ = torch.max(outputs, dim=1)
                    m = pred_confs

                    if scale:
                        target_confs = outputs[torch.arange(outputs.shape[0]), targets]
                        m = target_confs - pred_confs

                metrics.extend(m.tolist())

        return metrics

class MembershipInferenceAttack:
    def __init__(self, config: AttackConfig):
        self.config = config
        self.device = torch.device(f"cuda{config.gpu}" if torch.cuda.is_available() else "cpu")
        self.model_dir = self._get_model_directory()
        self.model, self.attack_loaders = self._initialize_model_and_data()
        self.extractor = ModelConfidenceExtractor()
        self.dataset_size = (len(self.attack_loaders[0].dataset.dataset)
                        if isinstance(self.attack_loaders[0].dataset, Subset)
                        else len(self.attack_loaders[0].dataset))

    def _get_model_directory(self) -> Path:
        dir_path = Path(MODEL_DIR) / self.config.exp_id
        if self.config.checkpoint:
            dir_path = dir_path / f'checkpoint_before_{self.config.checkpoint}'
        return dir_path

    def _initialize_model_and_data(self) -> Tuple[Module, DataLoader]:
        attack_loaders = [
            get_no_shuffle_train_loader(
                self.config.dataset,
                self.config.arch,
                self.config.batchsize,
                self.config.num_workers
            )
        ]

        if self.config.augment:
            attack_loaders.append(
                get_no_shuffle_train_loader(
                    self.config.dataset,
                    self.config.arch,
                    self.config.batchsize,
                    self.config.num_workers,
                    mirror_all=True)
            )

        model = load_model(
            self.config.arch,
            get_num_classes(self.config.dataset)
        ).to(self.device)

        return model, attack_loaders

    def _collect_shadow_model_data(self, method) -> Tuple[List[List[float]], List[List[int]]]:
        """Collect confidence scores from all shadow models."""
        shadow_confs = [[] for _ in self.attack_loaders] # augmentations x shadow_models x samples
        shadow_train_indices = []

        shadow_idx = 0
        while True:
            model_path = self.model_dir / f'shadow_{shadow_idx}'
            if not model_path.is_file():
                break

            saves = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(saves['model_state_dict'])

            for i, loader in enumerate(self.attack_loaders):
                shadow_confs[i].append(method(self.model, self.device, loader))

            shadow_train_indices.append(saves['trained_on_indices'])

            shadow_idx += 1

        shadow_confs = np.array(shadow_confs).transpose(1, 2, 0).tolist()

        return shadow_confs, shadow_train_indices

    def _compute_statistics(self, shadow_confs: List[List[float]],
                            shadow_train_indices: List[List[int]]) -> Dict:
        """Compute statistical measures for attack analysis."""
        all_indices = list(range(self.dataset_size))

        sample_in_confs = {i: [] for i in all_indices}
        sample_out_confs = {i: [] for i in all_indices}

        # Categorize confidences
        for confs, trained_indices in zip(shadow_confs, map(set, shadow_train_indices)):
            mask = np.isin(all_indices, list(trained_indices))
            for idx, conf, in_trained in zip(all_indices, confs, mask):
                target_dict = sample_in_confs if in_trained else sample_out_confs
                target_dict[idx].append(conf)

        if self.config.augment:
            f = lambda x: np.cov(x, rowvar=False)
        else:
            f = np.var

        # Compute statistics
        if len(shadow_confs) >= 64:
            in_var = [f(confs) for confs in sample_in_confs.values()]
            out_var = [f(confs) for confs in sample_out_confs.values()]
        else:
            global_in_var = f([c for confs in sample_in_confs.values() for c in confs])
            global_out_var = f([c for confs in sample_out_confs.values() for c in confs])
            in_var = [global_in_var] * self.dataset_size
            out_var = [global_out_var] * self.dataset_size

        in_means = [np.mean(confs, axis=0) for confs in sample_in_confs.values()]
        out_means = [np.mean(confs, axis=0) for confs in sample_out_confs.values()]

        return {
            'in_conf': sample_in_confs,
            'out_conf': sample_out_confs,
            'in_var': in_var,
            'out_var': out_var,
            'in_means': in_means,
            'out_means': out_means
        }

    def save_intermediate_results(self, stats: Dict, output_dir: str = 'logits/intermediate'):
        """Save intermediate statistical results."""
        save_dir = Path(STORAGE_DIR) / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        file_name = self.config.exp_id
        if self.config.checkpoint:
            file_name += f'_checkpoint_after_{self.config.checkpoint}'

        torch.save(stats, os.path.join(output_dir, f'{file_name}.pt'))

    def compute_intermediate_results(self):
        """Compute and save intermediate statistical results."""
        print("Computing intermediate results...")
        start_time = time.time()

        metric_dict = {"logits": self.extractor.get_logits,
                       "scaled_logits": self.extractor.get_scaled_logits,
                       "losses":self.extractor.get_losses}

        for i in metric_dict.keys():
            shadow_confs, shadow_indices = self._collect_shadow_model_data(metric_dict[i])
            logits_time = time.time()

            stats = self._compute_statistics(shadow_confs, shadow_indices)
            compute_time = time.time()

            self.save_intermediate_results(stats, output_dir='{i}_values/intermediate')



        print(f"Performance Timings:")
        print(f"Model loading: {logits_time - start_time:.2f}s")
        print(f"Logit computation: {compute_time - logits_time:.2f}s")
        print(f"Statistical analysis: {time.time() - compute_time:.2f}s")

    def _load_intermediate_stats(self, metric) -> pd.DataFrame:
        """Load intermediate statistical results from saved files.

        Returns:
            pd.DataFrame containing the previously computed statistics

        Raises:
            FileNotFoundError: If intermediate results file doesn't exist
        """
        stats_dir = Path(STORAGE_DIR) / f'{metric}_values/intermediate'

        file_name = self.config.exp_id
        if self.config.checkpoint:
            file_name += f'_checkpoint_after_{self.config.checkpoint}'

        stats_path = stats_dir / f'{file_name}.pt'

        if not stats_path.exists():
            raise FileNotFoundError(f"No intermediate results found at {stats_path}")

        return torch.load(stats_path)

    def _save_attack_results(self, scores: List[float], target_trained_on: List[int], output_dir: str):
        """Save attack results in CSV format with membership indicators."""
        # Create membership boolean array
        bools = [False] * len(self.attack_loader.dataset)
        for i, (_, _, idx) in enumerate(self.attack_loader.dataset):
            if idx in target_trained_on:
                bools[i] = True

        # Get original dataset length
        original_len = (len(self.attack_loaders[0].dataset.dataset)
                        if isinstance(self.attack_loaders[0].dataset, Subset)
                        else len(self.attack_loaders[0].dataset))

        # Generate all indices
        all_indices = list(range(original_len))

        # Create save directory
        save_dir = Path(STORAGE_DIR) / output_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        # Generate result filename
        result_name = (f'{self.config.exp_id}_{self.config.target_id}' if self.config.checkpoint is None
                       else f'{self.config.exp_id}_checkpoint_before_{self.config.checkpoint}_{self.config.target_id}')

        # Handle file collisions
        fullpath = save_dir / result_name
        if fullpath.exists():
            print("RESULTS EXIST BUT NOT OVERWRITING", file=sys.stderr)
        while fullpath.exists():
            fullpath = Path(str(fullpath) + "_")

        # Save results
        pd.DataFrame({
            f'{output_dir.rstrip("s")}_score': scores,
            'target_trained_on': bools,
            'og_idx': all_indices
        }).set_index('og_idx').to_csv(fullpath)


class LiRAAttack(MembershipInferenceAttack):
    def run(self):
        """Execute LiRA (Likelihood Ratio Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get target model confidences
        target_confs = self.extractor.get_scaled_logits(self.model, self.device, self.attack_loader)

        # Load intermediate results
        stats_df = self._load_intermediate_stats("scaled_logits")

        # Compute LiRA scores
        lira_scores = self._compute_lira_scores(target_confs, stats_df)

        # Save results
        self._save_attack_results(lira_scores, target_indices, 'lira')

    def _compute_lira_scores(self, target_confs: List[float], stats_df: pd.DataFrame) -> List[float]:
        """Compute LiRA scores using statistical analysis."""
        stats_df['in_std'] = np.sqrt(stats_df['in_var'])
        stats_df['out_std'] = np.sqrt(stats_df['out_var'])

        scores = []
        if self.config.augment:
            for i, conf in enumerate(target_confs):
                l = scipy.stats.multivariate_normal.pdf(conf, mean=stats_df['in_means'][i],
                                                        cov=stats_df['in_var'][i]) + 1e-64  # TODO
                r = scipy.stats.multivariate_normal.pdf(conf, mean=stats_df['out_means'][i], cov=stats_df['out_var'][i]) + 1e-64
                score = l / r
                scores.append(score)
        else:
            for i, conf in enumerate(target_confs):
                l = scipy.stats.norm.pdf(conf, loc=stats_df['in_means'][i], scale=stats_df['in_std'][i]) + 1e-64  # TODO
                r = scipy.stats.norm.pdf(conf, loc=stats_df['out_means'][i], scale=stats_df['out_std'][i]) + 1e-64
                score = l / r
                scores.append(score)

        return scores



class RMIAAttack(MembershipInferenceAttack):
    def run(self, gamma: float = 1.0):
        """Execute RMIA (Relative Membership Inference Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get confidences and compute ratios
        target_confs = self.extractor.get_logits(self.model, self.device, self.attack_loader)
        stats_df = self._load_intermediate_stats("logits")

        # Compute RMIA scores
        rmia_scores = self._compute_rmia_scores(target_confs, target_indices, stats_df, gamma)

        # Save results
        self._save_attack_results(rmia_scores, target_indices, 'rmia')

    def _compute_rmia_scores(self, target_confs: List[float], target_indices: List[int],
                             stats_df: pd.DataFrame, gamma: float) -> List[float]:
        """Compute RMIA scores using relative likelihood analysis."""
        test_indices = [i for i in range(self.dataset_size) if i not in target_indices]
        ratio_z = [target_confs[z] / np.mean(stats_df['out_conf'][z]) for z in test_indices]

        scores = []
        for i, conf in enumerate(target_confs):
            ratio_x = conf / (np.mean(stats_df['in_conf'][i] + stats_df['out_conf'][i]) / 2)
            scores.append(sum((ratio_x / r_z) > gamma for r_z in ratio_z))

        return scores

class AttackR(MembershipInferenceAttack):
    def run(self, alphas):
        """Execute RMIA (Relative Membership Inference Attack)."""
        # Load target model
        saves = torch.load(self.model_dir / self.config.target_id, map_location=self.device)
        self.model.load_state_dict(saves['model_state_dict'])
        target_indices = saves['trained_on_indices']

        # Get confidences and compute ratios
        target_confs = self.extractor.get_losses(self.model, self.device, self.attack_loader)
        stats_df = self._load_intermediate_stats("losses")

        # Save results
        for alpha in alphas:
            # Compute AttackR scores
            attackr_scores = self._compute_attackr_scores(target_confs, target_indices, stats_df, alpha)
            self._save_attack_results(attackr_scores, target_indices, 'attackr')

    @staticmethod
    def calculate_loss_threshold(alpha, distribution):
        threshold = np.quantile(distribution, q=alpha, interpolation='lower')
        return threshold

    @staticmethod
    def linear_itp_threshold_func(
            distribution: List[float],
            alpha: List[float],
            signal_min=0,
            signal_max=1000,
            **kwargs
    ) -> float:
        """
        Function that returns the threshold as the alpha quantile of
        a linear interpolation curve fit over the provided distribution.
        Args:
            distribution: Sequence of values that form the distribution from which
            the threshold is computed. (Here we only consider positive signal values.)
            alpha: Quantile value that will be used to obtain the threshold from the
                distribution.
            signal_min: minimum of all possible signal values, default value is zero
            signal_max: maximum of all possible signal values, default vaule is 1000
        Returns:
            threshold: alpha quantile of the provided distribution.
        """
        distribution = np.append(distribution, signal_min)
        distribution = np.append(distribution, signal_max)
        threshold = np.quantile(distribution, q=alpha, method='linear', **kwargs)
        return threshold

    def _compute_attackr_scores(self, target_confs: List[float], target_indices: List[int],
                             stats_df: pd.DataFrame, alpha: List[float]) -> List[float]:
        """Compute RMIA scores using relative likelihood analysis."""
        target_confs = self.extractor.get_logits(self.model, self.device, self.attack_loader)

        # for alpha in alphas:
        train_thresholds = []
        train_percs = []

        threshold_func = self.calculate_loss_threshold if alpha < 0.001 else self.linear_itp_threshold_func

        for point_idx, point_loss_dist in enumerate(target_confs):
            train_thresholds.append(threshold_func(alpha=alpha, distribution=point_loss_dist))
            train_percs.append(stats.percentileofscore(point_loss_dist, stats_df['out_conf'][point_idx]))

        return train_percs



def parse_args() -> AttackConfig:
    """Parse command line arguments and return attack configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batchsize', type=int, default=500)
    parser.add_argument('--target_id', default='target', type=str)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--attack', type=str, required=True, choices=['LiRA', 'RMIA', 'AttackR'])

    args = parser.parse_args()

    # Load saved configurations if available
    model_dir = Path(MODEL_DIR) / args.exp_id
    if args.checkpoint:
        model_dir = model_dir / f'checkpoint_before_{args.checkpoint}'

    try:
        saves = torch.load(model_dir / 'shadow_0')
        args.arch = args.arch or saves['arch']
        args.dataset = args.dataset or saves['dataset']
        args.augment = saves['hyperparameters']['augment']
    except Exception as e:
        print(f"Warning: Could not load all settings: {e}")

    return AttackConfig(
        exp_id=args.exp_id,
        target_id=args.target_id,
        checkpoint=args.checkpoint,
        arch=args.arch,
        dataset=args.dataset,
        augment=args.augment,
        batchsize=args.batchsize,
        num_workers=args.num_workers,
        gpu=args.gpu
    )


def main():
    config = parse_args()

    # Initialize and run appropriate attack
    if config.attack == 'LiRA':
        attack = LiRAAttack(config)
    elif config.attack == "RMIA":
        attack = RMIAAttack(config)
    else:  # RMIA
        attack = AttackR(config)

    # Compute intermediate results if they don't exist
    attack.compute_intermediate_results()
    attack.run()


if __name__ == "__main__":
    main()
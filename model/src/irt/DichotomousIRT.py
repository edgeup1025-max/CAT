from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mlflow
import matplotlib.pyplot as plt


class DichotomousIRT(BaseEstimator, RegressorMixin):
    """
    Trainable Dichotomous IRT model supporting 1PL (Rasch), 2PL, and 3PL.
    Joint-MLE approach: optimize person abilities (theta) and item params.
    """

    def __init__(
        self,
        n_items: int,
        model: str = "2PL",  # "1PL", "2PL", or "3PL"
        lr_ab: float = 1e-3,
        lr_theta: float = 5e-3,
        n_epochs: int = 1000,
        l2_lambda: float = 1e-4,
        patience: int = 50,
        clip_grad: float = 1.0,
        device: str = None,
        track_with_mlflow: bool = True,
    ):
        assert model in ("1PL", "2PL", "3PL")
        self.n_items = n_items
        self.model = model
        self.lr_ab = lr_ab
        self.lr_theta = lr_theta
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda
        self.patience = patience
        self.clip_grad = clip_grad
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.track_with_mlflow = track_with_mlflow

        if self.model == "1PL":
            self.alpha = None
        else:
            self.raw_alpha = nn.Parameter(torch.randn(n_items) * 0.5 + 0.5)
        self.b = nn.Parameter(torch.randn(n_items) * 0.5)

        if self.model == "3PL":
            self.raw_c = nn.Parameter(torch.full((n_items, ), -3.0))
        else:
            self.raw_c = None

        self._is_fitted = False

    def _positive(self, x):
        return F.softplus(x) + 1e-6

    def _guessing(self, raw_c):
        return torch.sigmoid(raw_c)

    def _prob(self, theta):
        if theta.dim() == 1:
            theta = theta.view(-1, 1)
        theta_exp = theta
        b_exp = self.b.view(1, self.n_items)

        if self.model == "1PL":
            a = torch.ones(self.n_items,
                           device=theta.device).view(1, self.n_items)
        else:
            a = self._positive(self.raw_alpha).view(1, self.n_items)

        logits = a * (theta_exp - b_exp)
        sigmoid = torch.sigmoid(logits.clamp(-20, 20))

        if self.model == "3PL":
            c = self._guessing(self.raw_c).view(1, self.n_items)
            P = c + (1.0 - c) * sigmoid
        else:
            P = sigmoid

        return torch.clamp(P, 1e-6, 1.0 - 1e-6)

    def fit(self, X, y=None, verbose=True):
        device = torch.device(self.device)
        X_np = np.asarray(X)
        assert X_np.ndim == 2 and X_np.shape[1] == self.n_items
        N = X_np.shape[0]

        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        thetas = nn.Parameter(torch.randn(N, device=device) * 0.5)

        self.b = nn.Parameter(self.b.detach().to(device))
        if self.model != "1PL":
            self.raw_alpha = nn.Parameter(self.raw_alpha.detach().to(device))
        if self.model == "3PL":
            self.raw_c = nn.Parameter(self.raw_c.detach().to(device))

        params = []
        params.append({'params': [self.b], 'lr': self.lr_ab})
        if self.model != "1PL":
            params.append({'params': [self.raw_alpha], 'lr': self.lr_ab})
        if self.model == "3PL":
            params.append({'params': [self.raw_c], 'lr': self.lr_ab})
        params.append({'params': [thetas], 'lr': self.lr_theta})

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.7,
                                                               patience=10)

        best_loss = float("inf")
        patience_counter = 0
        loss_history = []

        # === MLflow Setup ===
        if self.track_with_mlflow and mlflow.active_run():
            mlflow.log_params({
                "model_type": self.model,
                "n_items": self.n_items,
                "lr_ab": self.lr_ab,
                "lr_theta": self.lr_theta,
                "n_epochs": self.n_epochs,
                "l2_lambda": self.l2_lambda,
                "patience": self.patience,
                "device": str(self.device)
            })

        progress = tqdm(range(self.n_epochs),
                        desc="Training Dichotomous IRT",
                        unit="epoch",
                        disable=not verbose)
        for epoch in progress:
            optimizer.zero_grad()
            P = self._prob(thetas)
            nll = -(X_tensor * torch.log(P) +
                    (1 - X_tensor) * torch.log(1 - P)).sum() / (N *
                                                                self.n_items)

            l2 = (self.b**2).sum()
            if self.model != "1PL":
                l2 += (self._positive(self.raw_alpha)**2).sum()
            if self.model == "3PL":
                c = self._guessing(self.raw_c)
                l2 += (c**2).sum()
            l2 = self.l2_lambda * l2

            loss = nll + l2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [thetas, self.b] +
                ([self.raw_alpha] if self.model != "1PL" else []) +
                ([self.raw_c] if self.model == "3PL" else []),
                max_norm=self.clip_grad)
            optimizer.step()

            with torch.no_grad():
                thetas -= thetas.mean()

            scheduler.step(loss.item())

            loss_history.append(loss.item())
            progress.set_postfix({
                'loss': f"{loss.item():.6f}",
                'nll': f"{nll.item():.6f}"
            })

            # === MLflow Log Metrics ===
            if self.track_with_mlflow and mlflow.active_run():
                mlflow.log_metric("loss", loss.item(), step=epoch)
                mlflow.log_metric("nll", nll.item(), step=epoch)
                for i, pg in enumerate(optimizer.param_groups):
                    mlflow.log_metric(f"lr_group_{i}", pg["lr"], step=epoch)

            if loss.item() < best_loss - 1e-6:
                best_loss = loss.item()
                best_state = {
                    'thetas': thetas.detach().cpu().clone(),
                    'b': self.b.detach().cpu().clone(),
                }
                if self.model != "1PL":
                    best_state['raw_alpha'] = self.raw_alpha.detach().cpu(
                    ).clone()
                if self.model == "3PL":
                    best_state['raw_c'] = self.raw_c.detach().cpu().clone()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best
        with torch.no_grad():
            thetas_best = best_state['thetas'].to(device)
            self.b = nn.Parameter(best_state['b'].to(device))
            if self.model != "1PL":
                self.raw_alpha = nn.Parameter(
                    best_state['raw_alpha'].to(device))
            if self.model == "3PL":
                self.raw_c = nn.Parameter(best_state['raw_c'].to(device))

        # Final attributes
        self.thetas_ = thetas_best.cpu().numpy()
        self.b_ = self.b.detach().cpu().numpy()
        if self.model != "1PL":
            self.alpha_ = self._positive(self.raw_alpha).detach().cpu().numpy()
        else:
            self.alpha_ = np.ones(self.n_items)
        if self.model == "3PL":
            self.c_ = self._guessing(self.raw_c).detach().cpu().numpy()
        else:
            self.c_ = np.zeros(self.n_items)

        self._is_fitted = True
        progress.close()

        # === MLflow: Log Artifacts ===
        if self.track_with_mlflow and mlflow.active_run():
            # Log loss curve
            fig, ax = plt.subplots()
            ax.plot(loss_history, label="Training Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{self.model} IRT Loss Curve")
            ax.legend()
            mlflow.log_figure(fig, f"plots/{self.model}_loss_curve.png")
            plt.close(fig)

            # Log summary stats
            mlflow.log_metrics({
                "final_loss": float(best_loss),
                "alpha_mean": float(np.mean(self.alpha_)),
                "alpha_std": float(np.std(self.alpha_)),
                "b_mean": float(np.mean(self.b_)),
                "b_std": float(np.std(self.b_)),
                "c_mean": float(np.mean(self.c_)),
                "c_std": float(np.std(self.c_))
            })

        return self

    def predict_proba(self, X=None):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        thetas = torch.tensor(self.thetas_,
                              dtype=torch.float32,
                              device=self.b.device)
        P = self._prob(thetas)
        return P.detach().cpu().numpy()

    def predict_by_theta(self, thetas):
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted first.")
        thetas_t = torch.tensor(np.asarray(thetas).reshape(-1),
                                dtype=torch.float32,
                                device=self.b.device)
        P = self._prob(thetas_t)
        return P.detach().cpu().numpy()

    def expected_item_scores(self):
        return self.predict_proba()

    def expected_total_score(self):
        P = self.expected_item_scores()
        return P.sum(axis=1)

    def summary(self):
        if not self._is_fitted:
            print("Model not fitted.")
            return
        print("=== Dichotomous IRT Summary ===")
        print(f"Model: {self.model}")
        print(f"Items: {self.n_items}")
        print(
            f"Alpha mean={np.mean(self.alpha_):.3f}, std={np.std(self.alpha_):.3f}"
        )
        print(f"b mean={np.mean(self.b_):.3f}, std={np.std(self.b_):.3f}")
        if self.model == "3PL":
            print(f"c mean={np.mean(self.c_):.3f}, std={np.std(self.c_):.3f}")
        print(
            f"Theta mean={np.mean(self.thetas_):.3f}, std={np.std(self.thetas_):.3f}"
        )
        print("================================")

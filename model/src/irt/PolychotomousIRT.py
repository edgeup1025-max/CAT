from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import numpy as np


# ============================================================
#  GRM Loss (Negative Log-Likelihood)
# ============================================================
class GRMloss(nn.Module):
    """
    Graded Response Model Loss:
    Computes negative log-likelihood of observed responses given predicted probabilities.
    """

    def __init__(self, func):
        super().__init__()
        self.func = func  # Function that outputs P[batch, items, categories]

    def forward(self, X, thetas):
        P = self.func(
            thetas
        )  # Predicted probability distribution [N, items, categories]
        P = torch.clamp(P, min=1e-6, max=1.0)  # avoid log(0)

        # One-hot encode responses (integer-coded responses expected)
        X_int = torch.clamp(X.long(), 0, P.shape[2] - 1)
        X_onehot = F.one_hot(X_int, num_classes=P.shape[2]).float()

        # Negative log-likelihood
        nll = -(X_onehot * torch.log(P)).sum() / X_onehot.sum()

        # Stability check
        if torch.isnan(nll) or torch.isinf(nll):
            print(" NaN or Inf detected in GRM loss. Resetting to 0.")
            nll = torch.tensor(0.0, requires_grad=True, device=X.device)

        return nll


# ============================================================
#  Graded Response Model (GRM)
# ============================================================
class GRM(BaseEstimator, RegressorMixin):
    """
    PyTorch-based implementation of Samejima's Graded Response Model (GRM).
    Supports multi-category ordered responses (e.g., Likert 1–5).
    """

    def __init__(self,
                 n_items: int,
                 n_categories: int = 5,
                 learning_rate_alpha_beta: float = 1e-3,
                 learning_rate_theta: float = 5e-3,
                 n_epochs: int = 200,
                 l2_lambda: float = 1e-4,
                 patience: int = 30,
                 track_with_mlflow: bool = True):
        super().__init__()
        self.n_items = n_items
        self.n_categories = n_categories
        self.lr_ab = learning_rate_alpha_beta
        self.lr_theta = learning_rate_theta
        self.n_epochs = n_epochs
        self.l2_lambda = l2_lambda
        self.patience = patience
        self.track_with_mlflow = track_with_mlflow

        self.alpha = nn.Parameter(torch.randn(n_items) * 0.8 + 1.0)
        self.beta = nn.Parameter(
            torch.sort(torch.randn(n_items, n_categories - 1) * 1.0, dim=1)[0])
        self._is_fitted = False

    # ============================================================
    #  Vectorized GRM probability computation
    # ============================================================
    def _grm_probability_vectorized(self, thetas):
        """
        Compute P(X=k | θ) for all items and categories.
        thetas: [N]
        returns: [N, items, categories]
        """
        n_samples = thetas.shape[0]
        theta_exp = thetas.view(n_samples, 1, 1)
        alpha_exp = self.alpha.view(1, self.n_items, 1)
        beta_exp = self.beta.view(1, self.n_items, self.n_categories - 1)

        logits = alpha_exp * (theta_exp - beta_exp)
        P_star = torch.sigmoid(logits.clamp(-8, 8))

        ones = torch.ones((n_samples, self.n_items, 1), device=thetas.device)
        zeros = torch.zeros((n_samples, self.n_items, 1), device=thetas.device)
        P_star_full = torch.cat([ones, P_star, zeros], dim=2)

        # Compute category probabilities as difference of sigmoids
        P = P_star_full[:, :, :-1] - P_star_full[:, :, 1:]
        return torch.clamp(P, min=1e-6, max=1.0)

    # ============================================================
    #  Training (Fit)
    # ============================================================
    def fit(self, X, y=None):
        X_tensor = torch.tensor(X, dtype=torch.float32) - 1
        n_samples = X_tensor.shape[0]

        self.thetas = nn.Parameter(torch.randn(n_samples) * 0.5)

        optimizer = torch.optim.Adam([
            {
                "params": self.alpha,
                "lr": self.lr_ab
            },
            {
                "params": self.beta,
                "lr": self.lr_ab
            },
            {
                "params": self.thetas,
                "lr": self.lr_theta
            },
        ])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.7,
                                                               patience=10)

        criterion = GRMloss(func=self._grm_probability_vectorized)

        best_loss = float("inf")
        patience_counter = 0
        loss_history = []

        if self.track_with_mlflow:
            mlflow.log_params({
                "n_items": self.n_items,
                "n_categories": self.n_categories,
                "lr_alpha_beta": self.lr_ab,
                "lr_theta": self.lr_theta,
                "n_epochs": self.n_epochs,
                "l2_lambda": self.l2_lambda,
                "patience": self.patience
            })

        progress = tqdm(range(self.n_epochs),
                        desc="Training GRM",
                        unit="epoch")

        for epoch in progress:
            optimizer.zero_grad()
            loss = criterion(X_tensor, self.thetas)
            l2_reg = self.l2_lambda * (torch.sum(self.alpha**2) +
                                       torch.sum(self.beta**2))
            total_loss = loss + l2_reg
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [self.alpha, self.beta, self.thetas], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                self.beta[:] = torch.sort(self.beta, dim=1)[0]

            scheduler.step(total_loss.item())

            loss_history.append(total_loss.item())
            progress.set_postfix({"loss": f"{total_loss.item():.4f}"})

            if self.track_with_mlflow:
                mlflow.log_metric("train_loss", total_loss.item(), step=epoch)

            # Early stopping
            if total_loss.item() < best_loss - 1e-4:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        self._is_fitted = True
        self.thetas_ = self.thetas.detach().cpu().numpy()

        if self.track_with_mlflow:
            mlflow.log_param("best_loss", best_loss)
            mlflow.log_param("alpha_mean", float(self.alpha.mean().detach()))
            mlflow.log_param("alpha_std", float(self.alpha.std().detach()))
            mlflow.log_param("beta_mean", float(self.beta.mean().detach()))
            mlflow.log_param("beta_min", float(self.beta.min().detach()))
            mlflow.log_param("beta_max", float(self.beta.max().detach()))

            # Log loss curve as artifact
            import pandas as pd
            df = pd.DataFrame({
                "epoch": list(range(len(loss_history))),
                "loss": loss_history
            })
            df.to_csv(
                r"D:\WORKSPACE\OFIICE_WORKS\model\src\saver\grm_loss_curve.csv",
                index=False)
            mlflow.log_artifact(
                r"D:\WORKSPACE\OFIICE_WORKS\model\src\saver\grm_loss_curve.csv"
            )

        progress.close()
        return self

    # ============================================================
    #  Prediction: full category probabilities
    # ============================================================
    def predict(self):
        """
        Return full category probability tensor [N, items, categories].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        thetas_tensor = torch.tensor(self.thetas_, dtype=torch.float32)
        P = self._grm_probability_vectorized(thetas_tensor)
        return P.detach().cpu().numpy()

    # ============================================================
    #  Prediction: expected item-level scores
    # ============================================================
    def predict_expected_score(self):
        """
        Return expected score per item per respondent:
        E[X|θ] = sum_k k * P(X=k|θ)
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        thetas_tensor = torch.tensor(self.thetas_, dtype=torch.float32)
        P = self._grm_probability_vectorized(thetas_tensor)

        # Expected score computation
        category_indices = torch.arange(self.n_categories, device=P.device)
        expected_scores = (P * category_indices).sum(dim=2)
        return expected_scores.detach().cpu().numpy()

    # ============================================================
    #  Prediction: total (summed) score per respondent
    # ============================================================

    def predict_total_score(self):
        """
        Return total expected score per respondent:
        sum_i E[X_i|θ]
        """
        expected_item_scores = self.predict_expected_score()
        total_scores = expected_item_scores.sum(axis=1)
        return total_scores

    # ============================================================
    #  Model summary
    # ============================================================
    def summary(self):
        """
        Print and return GRM parameter summary.
        """
        if not self._is_fitted:
            print("Model not fitted yet.")
            return
        print("\n=== GRM Summary ===")
        print(f"Items: {self.n_items}")
        print(f"Categories: {self.n_categories}")
        print(
            f"Alpha (discrimination): mean={self.alpha.mean().item():.3f}, std={self.alpha.std().item():.3f}"
        )
        print(
            f"Beta (difficulty): mean={self.beta.mean().item():.3f}, range=({self.beta.min().item():.3f}, {self.beta.max().item():.3f})"
        )
        print(
            f"Theta (ability): mean={self.thetas.mean().item():.3f}, std={self.thetas.std().item():.3f}"
        )
        print("=====================\n")

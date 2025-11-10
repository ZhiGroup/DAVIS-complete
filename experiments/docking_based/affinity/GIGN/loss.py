import torch
import torch.nn.functional as F
from typing import Dict

class boltz2_loss(torch.nn.Module):
    """
    Loss for affinity prediction (regression only).

    Components
    ----------
    1) Absolute regression: Huber on predicted vs. true affinities.
    2) Pairwise regression: Huber on pairwise differences across the minibatch.

    Final loss = w_abs * L_abs + w_pair * L_pair

    Expected I/O
    ------------
    preds: {"affinity": float tensor (N,)}      # model predictions
    batch: {"y_aff":   float tensor (N,)}      # ground truth; NaN where missing
    """
    def __init__(
        self,
        w_abs: float = 1.0,
        w_pair: float = 2.0,
        huber_beta_abs: float = 0.5,
        huber_beta_pair: float = 0.5,
    ):
        super().__init__()
        self.w_abs = w_abs
        self.w_pair = w_pair
        self.beta_abs = huber_beta_abs
        self.beta_pair = huber_beta_pair

    @staticmethod
    def _mean_or_zero(x: torch.Tensor) -> torch.Tensor:
        return x.mean() if x.numel() > 0 else x.new_tensor(0.0)

    #def forward(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def forward(self, preds: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_hat = preds.view(-1)
        y_true = batch.view(-1)

        # 1) Absolute Huber loss
        """
            If y_true != 5 -> include in SmoothL1.
            If y_true == 5 and (y_hat - y_true) > 0 -> include in SmoothL1.
            Otherwise -> excluded (no loss).
        """
        L_abs = torch.tensor(0.0, device=y_hat.device)
        resid = y_hat - y_true
        mask_abs = (y_true != 5) | ((y_true == 5) & (resid > 0))

        if mask_abs.any():
            L_abs = F.smooth_l1_loss(y_hat[mask_abs], y_true[mask_abs], beta=self.beta_abs, reduction="mean")
        else:
            L_abs = y_hat.new_tensor(0.0)

        # 2) All-pairs Huber loss across minibatch
        
        p = y_hat   
        t = y_true  
        M = p.shape[0]

        # Pairwise matrices
        pi = p.unsqueeze(1)                  # (M,1)
        pj = p.unsqueeze(0)                  # (1,M)
        ti = t.unsqueeze(1)                  # (M,1)
        tj = t.unsqueeze(0)                  # (1,M)

        d_pred = pi - pj                     # (M,M)
        d_true = ti - tj                     # (M,M)

        # Conditions
        gt5_i = ti > 5
        gt5_j = tj > 5
        eq5_i = ti == 5
        eq5_j = tj == 5

        # 1) y_true_i > 5 and y_true_j > 5
        cond1 = gt5_i & gt5_j
        # 2) y_true_i > 5 and y_true_j = 5, and (y_pred_i - y_pred_j) < 0
        cond2 = gt5_i & eq5_j & (d_pred < 0)
        # 3) y_true_i = 5 and y_true_j > 5, and (y_pred_i - y_pred_j) > 0
        cond3 = eq5_i & gt5_j & (d_pred > 0)
        # 4) y_true_i = 5 and y_true_j = 5  -> EXCLUDED (no gradient)

        pair_mask_full = cond1 | cond2 | cond3

        # Use each unordered pair once (i < j)
        triu = torch.triu(torch.ones((M, M), dtype=torch.bool, device=p.device), diagonal=1)
        sel = pair_mask_full & triu

        if sel.any():
            diff_pair = (d_pred - d_true)[sel]   # ((#selected_pairs),)
            L_pair = F.smooth_l1_loss(
                diff_pair, torch.zeros_like(diff_pair),
                beta=self.beta_pair, reduction="mean"
            )
        else:
            L_abs = y_hat.new_tensor(0.0)

        loss = self.w_abs * L_abs + self.w_pair * L_pair
        return loss

# -------------------------
# Minimal usage example
# -------------------------
if __name__ == "__main__":

    y_true = torch.tensor([6.0, 7.0, 5.0, 5.0, 6.0, 8.0])
    y_hat  = torch.tensor([1.0, 2.0, 3.0, 8.1, 10.0, 2.5], requires_grad=True)
    

    criterion = boltz2_loss(
        w_abs=0.5, w_pair=0.5,
        huber_beta_abs=0.5, huber_beta_pair=0.5
    )
    out = criterion(y_hat, y_true)
    print(out)

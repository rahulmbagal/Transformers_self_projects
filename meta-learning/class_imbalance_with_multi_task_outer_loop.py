# ==============================================
# Phase 3: First-Order Meta-Learning (Reptile) in PyTorch
# Updates:
#  - Inner loop: per-task class-weighted BCE for imbalance
#  - Outer loop: imbalance-aware weighted average over tasks
#  - Sampler: option to create balanced or imbalanced tasks
#  - NEW: Group 1 multi-task cold-start evaluation by segment (no leakage)
# ==============================================

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
import random
import pandas as pd  # NEW (for nice summary table)

# -------------------------------
# 0) Repro + Device
# -------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 1) Load processed Phase 2 data
# -------------------------------
bundle = joblib.load("phase2_processed_data.pkl")

X2_train, y2_train = bundle["group2_train"]
X2_val,   y2_val   = bundle["group2_val"]
X2_test,  y2_test  = bundle["group2_test"]

X1_train, y1_train = bundle["group1_train"]
X1_val,   y1_val   = bundle["group1_val"]
X1_test,  y1_test  = bundle["group1_test"]

# Optional: per-row task IDs for Group 1 (e.g., 'SAP_PRODUCT_LINE' per row)
task_ids_train = bundle.get("group1_task_ids_train", None)
task_ids_val   = bundle.get("group1_task_ids_val", None)
task_ids_test  = bundle.get("group1_task_ids_test", None)

# Convert labels to numpy (float32)
to_f32 = lambda a: np.asarray(a).astype(np.float32)
y2_train, y2_val, y2_test = map(to_f32, [y2_train, y2_val, y2_test])
y1_train, y1_val, y1_test = map(to_f32, [y1_train, y1_val, y1_test])

input_dim = X2_train.shape[1]

# -------------------------------
# 2) Small helpers
# -------------------------------
def np_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)

@torch.no_grad()
def evaluate_numpy(model, X_np, y_np, batch=4096):
    model.eval()
    n = len(y_np)
    total_loss, probs = 0.0, []
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for i in range(0, n, batch):
        Xb = torch.tensor(X_np[i:i+batch], dtype=torch.float32, device=device)
        yb = torch.tensor(y_np[i:i+batch].reshape(-1,1), dtype=torch.float32, device=device)
        logit = model(Xb)
        total_loss += bce(logit, yb).item()
        probs.append(torch.sigmoid(logit).squeeze(1).cpu().numpy())
    probs = np.concatenate(probs)
    return total_loss / n, np_auc(y_np, probs)

# Per-task, per-sample weighted BCE to fix imbalance in INNER loop
def weighted_bce_for_task(y_tensor):
    """
    y_tensor: shape [N,1] with {0,1}
    Returns nn.BCEWithLogitsLoss(weight=per_sample_weights)
    Weights are computed so positives and negatives contribute ~equally,
    and the average weight ~ 1 (divide by 2 trick).
    """
    y_flat = y_tensor.view(-1)
    pos = (y_flat == 1).float().sum()
    neg = (y_flat == 0).float().sum()
    total = pos + neg
    if total.item() == 0 or pos.item() == 0 or neg.item() == 0:
        # Degenerate: fallback to unweighted
        return nn.BCEWithLogitsLoss()

    w_pos = total / (2.0 * pos)
    w_neg = total / (2.0 * neg)
    weights = torch.where(y_flat == 1, w_pos, w_neg).view(-1, 1)
    return nn.BCEWithLogitsLoss(weight=weights)

# Task-level weight for OUTER loop (penalize imbalanced query sets)
def task_weight_from_query(yq, alpha=2.0, use_size_weight=False):
    """
    yq: tensor of shape [Nq,1] with {0,1}
    alpha: strength of imbalance penalty (higher => stronger down-weighting)
    use_size_weight: if True, also multiply by task size (stabilizes gradients)
    """
    y_flat = yq.view(-1)
    pos = (y_flat == 1).sum().item()
    neg = (y_flat == 0).sum().item()
    total = pos + neg
    if total == 0:
        return 1.0
    imbalance_ratio = abs(pos - neg) / total  # 0 balanced … 1 skewed
    w = float(np.exp(-alpha * imbalance_ratio))  # in (0,1], 1 if perfectly balanced
    if use_size_weight:
        w *= total
    return w

# -------------------------------
# 3) Model
# -------------------------------
class TabMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), p=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # logits
        return self.net(x)

# -------------------------------
# 4) Task sampler
# -------------------------------
@dataclass
class TaskBatch:
    Xs: torch.Tensor  # [n_tasks, n_support, D]
    ys: torch.Tensor  # [n_tasks, n_support, 1]
    Xq: torch.Tensor  # [n_tasks, n_query, D]
    yq: torch.Tensor  # [n_tasks, n_query, 1]

def sample_task_batch(
    X, y,
    n_tasks=8,
    k_support_per_class=8,
    k_query_per_class=16,
    balanced=True,
    pos_label=1.0,
    neg_label=0.0
):
    X = np.asarray(X); y = np.asarray(y)
    idx_pos = np.where(y == pos_label)[0]
    idx_neg = np.where(y == neg_label)[0]

    tasks_Xs, tasks_ys, tasks_Xq, tasks_yq = [], [], [], []

    for _ in range(n_tasks):
        if balanced and len(idx_pos) > 0 and len(idx_neg) > 0:
            # Class-balanced support/query
            sup_pos = np.random.choice(idx_pos, size=k_support_per_class, replace=True)
            sup_neg = np.random.choice(idx_neg, size=k_support_per_class, replace=True)
            qry_pos = np.random.choice(idx_pos, size=k_query_per_class, replace=True)
            qry_neg = np.random.choice(idx_neg, size=k_query_per_class, replace=True)

            Xs = np.vstack([X[sup_pos], X[sup_neg]])
            ys = np.concatenate([np.ones(k_support_per_class), np.zeros(k_support_per_class)])[:, None]
            Xq = np.vstack([X[qry_pos], X[qry_neg]])
            yq = np.concatenate([np.ones(k_query_per_class), np.zeros(k_query_per_class)])[:, None]
        else:
            # Imbalanced (natural) sampling
            n_sup = 2 * k_support_per_class
            n_qry = 2 * k_query_per_class
            sup_idx = np.random.choice(len(y), size=n_sup, replace=True)
            qry_idx = np.random.choice(len(y), size=n_qry, replace=True)
            Xs = X[sup_idx]; ys = y[sup_idx][:, None]
            Xq = X[qry_idx]; yq = y[qry_idx][:, None]

        # Shuffle within each split
        s_idx = np.random.permutation(Xs.shape[0])
        q_idx = np.random.permutation(Xq.shape[0])
        tasks_Xs.append(Xs[s_idx]); tasks_ys.append(ys[s_idx])
        tasks_Xq.append(Xq[q_idx]); tasks_yq.append(yq[q_idx])

    # Stack (uniform sizes across tasks)
    Xs = torch.tensor(np.stack(tasks_Xs), dtype=torch.float32, device=device)
    ys = torch.tensor(np.stack(tasks_ys), dtype=torch.float32, device=device)
    Xq = torch.tensor(np.stack(tasks_Xq), dtype=torch.float32, device=device)
    yq = torch.tensor(np.stack(tasks_yq), dtype=torch.float32, device=device)
    return TaskBatch(Xs, ys, Xq, yq)

# -------------------------------
# 5) Param vector utils (Reptile)
# -------------------------------
def get_param_vector(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def set_param_vector_(model, vec):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[idx: idx+n].view_as(p))
        idx += n

def clone_param_vector(model):
    return torch.cat([p.detach().clone().flatten() for p in model.parameters()])

# -------------------------------
# 6) Inner loop with per-task weighted BCE
# -------------------------------
def inner_train_one_task(model, Xs, ys, inner_steps=5, inner_lr=1e-2):
    """
    Returns parameter vector AFTER adapting on the task's support set.
    Uses weighted BCE to correct class imbalance inside the task.
    """
    theta = clone_param_vector(model)
    temp_model = TabMLP(input_dim).to(device)
    set_param_vector_(temp_model, theta)

    opt = torch.optim.SGD(temp_model.parameters(), lr=inner_lr)
    loss_fn = weighted_bce_for_task(ys)  # <-- imbalance-aware

    for _ in range(inner_steps):
        opt.zero_grad()
        logits = temp_model(Xs)
        loss = loss_fn(logits, ys)
        loss.backward()
        opt.step()

    return get_param_vector(temp_model)  # θ'

# -------------------------------
# 7) Reptile meta-training with OUTER weighting
# -------------------------------
def reptile_meta_train(
    X_train_np, y_train_np,
    X_val_np,   y_val_np,
    input_dim,
    meta_iters=1000,
    n_tasks=8,
    k_support_per_class=8,
    k_query_per_class=16,
    inner_steps=5,
    inner_lr=1e-2,
    meta_lr=5e-2,
    eval_every=50,
    balanced_tasks=True,      # set False to allow imbalanced tasks
    outer_alpha=2.0,          # imbalance penalty strength for outer loop
    outer_use_size_weight=False  # also weight by query size if True
):
    model = TabMLP(input_dim).to(device)
    base_theta = get_param_vector(model)

    best_val_auc = -np.inf
    best_state = clone_param_vector(model)

    for it in range(1, meta_iters + 1):
        # 1) Sample tasks (balanced or imbalanced)
        tasks = sample_task_batch(
            X_train_np, y_train_np,
            n_tasks=n_tasks,
            k_support_per_class=k_support_per_class,
            k_query_per_class=k_query_per_class,
            balanced=balanced_tasks
        )

        # 2) Adapt on each task's support -> θ'_t, also compute task weights from query labels
        adapted_thetas = []
        weights = []
        for t in range(n_tasks):
            Xs, ys = tasks.Xs[t], tasks.ys[t]
            Xq, yq = tasks.Xq[t], tasks.yq[t]

            theta_prime = inner_train_one_task(model, Xs, ys,
                                               inner_steps=inner_steps, inner_lr=inner_lr)
            adapted_thetas.append(theta_prime)

            # imbalance-aware task weight (outer loop)
            w_t = task_weight_from_query(yq, alpha=outer_alpha, use_size_weight=outer_use_size_weight)
            weights.append(w_t)

        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        weights = torch.clamp(weights, min=1e-8)  # safety

        # 3) Weighted average of adapted params: avg_theta' = sum(w_t θ'_t) / sum(w_t)
        adapted_stack = torch.stack(adapted_thetas, dim=0)  # [n_tasks, P]
        w = (weights / weights.sum()).view(-1, 1)
        avg_theta_prime = (w * adapted_stack).sum(dim=0)

        # 4) Reptile meta-update: θ ← θ + β (avg_theta' - θ)
        with torch.no_grad():
            base_theta = base_theta + meta_lr * (avg_theta_prime - base_theta)
            set_param_vector_(model, base_theta)

        # 5) Periodic validation on Group 2
        if it % eval_every == 0:
            val_loss, val_auc = evaluate_numpy(model, X_val_np, y_val_np)
            print(f"[Meta-iter {it:04d}] Group2 Val: loss={val_loss:.4f} AUC={val_auc:.4f}")
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = clone_param_vector(model)

    set_param_vector_(model, best_state)
    print(f"✅ Meta-training done. Best Group2 Val AUC: {best_val_auc:.4f}")
    return model

# -------------------------------
# 8) Run meta-training on Group 2
# -------------------------------
meta_model = reptile_meta_train(
    X2_train, y2_train,
    X2_val,   y2_val,
    input_dim=input_dim,
    meta_iters=600,
    n_tasks=8,
    k_support_per_class=8,
    k_query_per_class=16,
    inner_steps=5,
    inner_lr=1e-2,
    meta_lr=5e-2,
    eval_every=50,
    balanced_tasks=False,        # allow natural (possibly imbalanced) tasks
    outer_alpha=2.0,             # stronger penalty → smaller weight for skewed tasks
    outer_use_size_weight=False  # set True to weight by query size *and* balance
)

# Quick check on Group 2 test
g2_test_loss, g2_test_auc = evaluate_numpy(meta_model, X2_test, y2_test)
print(f"Group 2 (rich) Test: loss={g2_test_loss:.4f} AUC={g2_test_auc:.4f}")

# -------------------------------
# 9) Single-task adaptation on Group 1 (few-shot)
# -------------------------------
def adapt_to_group1(
    meta_model,
    X1_train_np, y1_train_np,
    X1_val_np,   y1_val_np,
    shots_per_class=32,
    steps=100,
    lr=1e-3,
    freeze_until=1  # freeze first Linear layer; try None/0/1/2
):
    y = np.asarray(y1_train_np)
    pos_idx = np.where(y == 1.0)[0]
    neg_idx = np.where(y == 0.0)[0]

    sup_pos = np.random.choice(pos_idx, size=shots_per_class, replace=(len(pos_idx) < shots_per_class))
    sup_neg = np.random.choice(neg_idx, size=shots_per_class, replace=(len(neg_idx) < shots_per_class))

    X_sup = np.vstack([X1_train_np[sup_pos], X1_train_np[sup_neg]])
    y_sup = np.concatenate([np.ones(shots_per_class), np.zeros(shots_per_class)])[:, None]

    idx = np.random.permutation(len(y_sup))
    X_sup, y_sup = X_sup[idx], y_sup[idx]

    # Clone meta model params
    model = TabMLP(input_dim).to(device)
    set_param_vector_(model, get_param_vector(meta_model))

    # Optional freezing
    if freeze_until is not None:
        depth = 0
        for m in model.net:
            if isinstance(m, nn.Linear):
                if depth < freeze_until:
                    for p in m.parameters():
                        p.requires_grad = False
                depth += 1

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # Use weighted BCE even in adaptation (Group 1 might be imbalanced)
    y_sup_t = torch.tensor(y_sup, dtype=torch.float32, device=device)
    X_sup_t = torch.tensor(X_sup, dtype=torch.float32, device=device)
    loss_fn = weighted_bce_for_task(y_sup_t)

    model.train()
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X_sup_t)
        loss = loss_fn(logits, y_sup_t)
        loss.backward()
        opt.step()

    # Eval
    val_loss, val_auc = evaluate_numpy(model, X1_val_np, y1_val_np)
    test_loss, test_auc = evaluate_numpy(model, X1_test, y1_test)
    return model, (val_loss, val_auc), (test_loss, test_auc)

adapted_model, (g1_val_loss, g1_val_auc), (g1_test_loss, g1_test_auc) = adapt_to_group1(
    meta_model,
    X1_train, y1_train,
    X1_val,   y1_val,
    shots_per_class=32,
    steps=100,
    lr=1e-3,
    freeze_until=1
)

print(f"Group 1 (cold)  Val:  loss={g1_val_loss:.4f} AUC={g1_val_auc:.4f}")
print(f"Group 1 (cold)  Test: loss={g1_test_loss:.4f} AUC={g1_test_auc:.4f}")

# -------------------------------
# 9b) NEW: Multi-task cold-start adaptation on Group 1 segments (no leakage)
#       - Support is drawn from Group1 TRAIN only (per segment)
#       - Eval is on that segment's VAL+TEST only
# -------------------------------
def evaluate_group1_multitask(
    meta_model,
    X1_train_np, y1_train_np, task_ids_train,
    X1_val_np,   y1_val_np,   task_ids_val,
    X1_test_np,  y1_test_np,  task_ids_test,
    shots_per_class=32,
    steps=100,
    lr=1e-3,
    freeze_until=1
):
    # Fallback: if task IDs not provided, create demo buckets (replace with real metadata)
    if task_ids_train is None or task_ids_val is None or task_ids_test is None:
        print("⚠️  No Group1 task IDs found in bundle. Creating demo task IDs (3 buckets). Replace with real metadata.")
        rng = np.random.default_rng(SEED)
        task_ids_train = rng.choice(["TaskA","TaskB","TaskC"], size=len(y1_train_np))
        task_ids_val   = rng.choice(["TaskA","TaskB","TaskC"], size=len(y1_val_np))
        task_ids_test  = rng.choice(["TaskA","TaskB","TaskC"], size=len(y1_test_np))

    # Unique tasks appearing anywhere in Group 1
    unique_tasks = np.unique(np.concatenate([np.asarray(task_ids_train),
                                             np.asarray(task_ids_val),
                                             np.asarray(task_ids_test)]))

    results = []
    for task in unique_tasks:
        # Masks per split for this task
        m_tr = (np.asarray(task_ids_train) == task)
        m_va = (np.asarray(task_ids_val)   == task)
        m_te = (np.asarray(task_ids_test)  == task)

        X_tr_t, y_tr_t = X1_train_np[m_tr], y1_train_np[m_tr]
        X_ev_t = np.vstack([X1_val_np[m_va], X1_test_np[m_te]])
        y_ev_t = np.concatenate([y1_val_np[m_va], y1_test_np[m_te]])

        # Need both classes for few-shot support & eval
        pos_idx = np.where(y_tr_t == 1.0)[0]
        neg_idx = np.where(y_tr_t == 0.0)[0]
        if len(pos_idx) < 5 or len(neg_idx) < 5 or len(np.unique(y_ev_t)) < 2:
            print(f"Skipping task '{task}': insufficient class balance in train or eval.")
            continue

        k = min(shots_per_class, len(pos_idx), len(neg_idx))
        sup_pos = np.random.choice(pos_idx, size=k, replace=(len(pos_idx) < k))
        sup_neg = np.random.choice(neg_idx, size=k, replace=(len(neg_idx) < k))
        X_sup = np.vstack([X_tr_t[sup_pos], X_tr_t[sup_neg]])
        y_sup = np.concatenate([np.ones(k), np.zeros(k)])[:, None]

        # Clone & (optionally) freeze
        model = TabMLP(input_dim).to(device)
        set_param_vector_(model, get_param_vector(meta_model))

        if freeze_until is not None:
            depth = 0
            for m in model.net:
                if isinstance(m, nn.Linear):
                    if depth < freeze_until:
                        for p in m.parameters():
                            p.requires_grad = False
                    depth += 1

        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        X_sup_t = torch.tensor(X_sup, dtype=torch.float32, device=device)
        y_sup_t = torch.tensor(y_sup, dtype=torch.float32, device=device)
        loss_fn = weighted_bce_for_task(y_sup_t)

        model.train()
        for _ in range(steps):
            opt.zero_grad()
            logits = model(X_sup_t)
            loss = loss_fn(logits, y_sup_t)
            loss.backward()
            opt.step()

        # Evaluate on VAL+TEST for this task only
        model.eval()
        with torch.no_grad():
            X_eval_t = torch.tensor(X_ev_t, dtype=torch.float32, device=device)
            y_eval_t = torch.tensor(y_ev_t[:, None], dtype=torch.float32, device=device)
            logits_eval = model(X_eval_t)
            preds = torch.sigmoid(logits_eval).squeeze(1).cpu().numpy()
            auc = roc_auc_score(y_ev_t, preds) if len(np.unique(y_ev_t)) > 1 else np.nan
            loss_eval = nn.BCEWithLogitsLoss(reduction="mean")(logits_eval, y_eval_t).item()

        print(f"[Group1 task '{task}'] eval_loss={loss_eval:.4f} AUC={auc:.4f}")
        results.append({"task": str(task), "eval_loss": loss_eval, "eval_auc": auc})

    if not results:
        print("No valid Group 1 tasks to evaluate.")
        return None

    df = pd.DataFrame(results).sort_values("eval_auc", ascending=False)
    print("\n===== Group 1 multi-task cold-start summary =====")
    print(df)
    print("-----------------------------------------------")
    print(f"Mean AUC across tasks: {df['eval_auc'].mean():.4f}")
    print(f"Std  AUC across tasks: {df['eval_auc'].std():.4f}")
    return df

# Run multi-task evaluation on Group 1 segments
_ = evaluate_group1_multitask(
    meta_model,
    X1_train, y1_train, task_ids_train,
    X1_val,   y1_val,   task_ids_val,
    X1_test,  y1_test,  task_ids_test,
    shots_per_class=32,
    steps=100,
    lr=1e-3,
    freeze_until=1
)

# -------------------------------
# 10) Save adapted model (optional)
# -------------------------------
torch.save({
    "state_dict": adapted_model.state_dict(),
    "input_dim": input_dim
}, "meta_adapted_group1.pt")

print("✅ Phase 3 complete: Reptile with imbalance-aware inner/outer loops + Group1 multi-task cold-start evaluation.")

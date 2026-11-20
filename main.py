import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False
VAL_FRAC = 0.2
SEED = 0
W_GRID = np.linspace(0.0, 0.95, 20)
T_GRID = [0.8, 1.0, 1.25, 1.5, 2.0]
MLP_W_GRID = [0.0, 0.2, 0.35, 0.5, 0.7]
MLP_MIN_GAIN = 0.005 
MAX_CAT = 30
TIME_LIMIT = 600


# --- Ablation switches (flip True/False to benchmark each upgrade for your report) ---
USE_MLP = True
USE_GIVEN_GATE = True
USE_PREDICT_GATE = True

PARAMS = dict(learning_rate=0.03, num_leaves=16, min_child_samples=40,
              reg_lambda=5.0, verbose=-1)

CAT_ITERS = 350
CAT_LR = 0.20
CAT_DEPTH = 4
CAT_THREADS = -1

MLP_HIDDEN = 256
MLP_DROPOUT = 0.25
MLP_EPOCHS = 60
MLP_PATIENCE = 6
MLP_BS = 256


# ------------------------------------------------------------------ calibration

def temper(p, T):
    q = np.clip(p, 1e-12, None) ** (1.0 / T)
    return q / q.sum(axis=1, keepdims=True)


def blend(p, marginal, W, T):
    return W * temper(p, T) + (1.0 - W) * marginal


def nll_1d(p, y):
    ok = y >= 0
    if not ok.any():
        return 0.0
    py = np.clip(p[ok, y[ok]], 1e-12, 1.0)
    return float(-np.log(py).mean())


# ----------------------------------------------------------- multi-task PyTorch MLP

class SurveyMLP(nn.Module):
    """Shared-trunk MLP predicting all PREDICT items simultaneously."""

    def __init__(self, d_in, P, K, log_marg, valid_mask,
                 hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.W = nn.Parameter(torch.randn(P, hidden, K) / (hidden ** 0.5) * 0.5)
        self.b = nn.Parameter(log_marg.clone())
        self.register_buffer("addmask", torch.where(valid_mask, 0.0, -1e9))

    def forward(self, x):
        h = self.body(x)
        return torch.einsum("bd,jdk->bjk", h, self.W) + self.b + self.addmask


def build_onehot_given(frame, given, options_fn):
    """Builds a dense float32 one-hot matrix from GIVEN columns (missing -> extra slot)."""
    n = len(frame)
    widths = [len(options_fn(c)) + 1 for c in given]
    D = sum(widths)
    X_oh = np.zeros((n, max(D, 1)), dtype=np.float32)
    off = 0
    rows = np.arange(n)
    for c, w in zip(given, widths):
        opts_c = options_fn(c)
        codes = pd.Categorical(frame[c], categories=opts_c).codes.astype(np.int64)
        idx = np.where(codes >= 0, codes, w - 1)
        X_oh[rows, off + idx] = 1.0
        off += w
    return X_oh


def _mlp_nll(logits, y):
    lp = F.log_softmax(logits, dim=-1)
    m = y >= 0
    g = lp.gather(2, y.clamp(min=0)[..., None])[..., 0]
    return -(g * m).sum() / m.sum().clamp(min=1)


def fit_predict_mlp(X_oh, Y_mat, k_sizes, tr_pos, va_pos, pred_pos,
                    epochs=MLP_EPOCHS, seed=SEED):
    """Fits SurveyMLP on tr_pos (early stopping on va_pos if provided) and predicts on va_pos & pred_pos."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P, K = len(k_sizes), int(max(k_sizes))

    valid = np.arange(K)[None, :] < np.asarray(k_sizes)[:, None]
    marg = np.zeros((P, K), dtype=np.float32)
    for j, kj in enumerate(k_sizes):
        y = Y_mat[tr_pos, j]
        cnt = np.bincount(y[y >= 0], minlength=kj).astype(np.float32) + 0.5
        marg[j, :kj] = cnt / cnt.sum()
    log_marg = torch.as_tensor(
        np.where(valid, np.log(np.maximum(marg, 1e-9)), 0.0),
        dtype=torch.float32, device=dev)
    valid_t = torch.as_tensor(valid, device=dev)

    Xt = torch.as_tensor(X_oh, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor(Y_mat, dtype=torch.long, device=dev)
    tr_t = torch.as_tensor(tr_pos, dtype=torch.long, device=dev)
    va_t = torch.as_tensor(va_pos, dtype=torch.long, device=dev) if len(va_pos) else None

    model = SurveyMLP(X_oh.shape[1], P, K, log_marg, valid_t).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    best_loss, best_state, stale, best_ep = float("inf"), None, 0, epochs
    n_tr = len(tr_t)
    for ep in range(1, epochs + 1):
        model.train()
        perm = tr_t[torch.randperm(n_tr, device=dev)]
        for s in range(0, n_tr, MLP_BS):
            b = perm[s:s + MLP_BS]
            loss = _mlp_nll(model(Xt[b]), Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        if va_t is not None:
            model.eval()
            with torch.no_grad():
                v_loss = _mlp_nll(model(Xt[va_t]), Yt[va_t]).item()
            if v_loss < best_loss - 1e-4:
                best_loss = v_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_ep = ep
                stale = 0
            else:
                stale += 1
                if stale >= MLP_PATIENCE:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    @torch.no_grad()
    def _pred(rows):
        if len(rows) == 0:
            return np.zeros((0, P, K), dtype=np.float32)
        model.eval()
        rt = torch.as_tensor(rows, dtype=torch.long, device=dev)
        out = [F.softmax(model(Xt[rt[s:s + 4096]]), dim=-1)
               for s in range(0, len(rt), 4096)]
        return torch.cat(out, dim=0).cpu().numpy()

    return _pred(va_pos), _pred(pred_pos), best_ep


# ----------------------------------------------------------- skip-logic gating

def build_given_gate_mask(frame, schema, target, opts_t, tr_pos, Y_t):
    """Returns (is_open, known) boolean arrays if target has a verified GIVEN gate, else None."""
    if not USE_GIVEN_GATE:
        return None
    gate = schema["items"][target].get("gate")
    if not gate:
        return None
    parent = gate.get("parent")
    if schema["items"].get(parent, {}).get("class") != "GIVEN":
        return None

    obs_raw = set(gate["observed_if"])
    obs_str = {str(v) for v in gate["observed_if"]}
    col = frame[parent].to_numpy(dtype=object)
    known = ~pd.isna(col)
    is_open = np.array([
        (v in obs_raw or str(v) in obs_str) if k else False
        for v, k in zip(col, known)
    ], dtype=bool)

    sent = len(opts_t) - 1
    y_tr = Y_t[tr_pos]
    ok = y_tr >= 0
    o_tr = is_open[tr_pos] & known[tr_pos] & ok
    c_tr = (~is_open[tr_pos]) & known[tr_pos] & ok
    if o_tr.sum() == 0 or c_tr.sum() == 0:
        return None
    if (y_tr[o_tr] != sent).mean() < 0.995 or (y_tr[c_tr] == sent).mean() < 0.995:
        return None
    return is_open, known


def apply_given_gate(p_rows, row_pos, gate_info):
    """Zeros out options ruled out by a verified GIVEN gate and renormalizes."""
    if gate_info is None or len(row_pos) == 0:
        return p_rows
    is_open, known = gate_info
    p = p_rows.copy()
    sent = p.shape[1] - 1
    o_r = is_open[row_pos] & known[row_pos]
    c_r = (~is_open[row_pos]) & known[row_pos]

    if o_r.any():
        p[o_r, sent] = 0.0
        s = p[o_r].sum(axis=1, keepdims=True)
        fallback = np.zeros_like(p[o_r])
        fallback[:, :sent] = 1.0 / max(sent, 1)
        p[o_r] = np.where(s > 1e-9, p[o_r] / np.maximum(s, 1e-30), fallback)
    if c_r.any():
        p[c_r, :] = 0.0
        p[c_r, sent] = 1.0

    p = np.clip(p, 1e-9, None)
    return p / p.sum(axis=1, keepdims=True)


def fit_parent_transition(Y_child, Y_parent, fit_pos, K_child, K_par):
    """Empirical P(child is asked | parent option k) with Laplace smoothing."""
    sent = K_child - 1
    yc, yp = Y_child[fit_pos], Y_parent[fit_pos]
    ok = (yc >= 0) & (yp >= 0)
    if not ok.any():
        return None
    asked = (yc[ok] != sent).astype(float)
    tot = np.bincount(yp[ok], minlength=K_par).astype(float)
    ask = np.bincount(yp[ok], weights=asked, minlength=K_par)
    return (ask + 0.5) / (tot + 1.0)


def apply_parent_transition(p_child, p_parent, T_table):
    """Combines P(asked) = p_parent @ T_table with renormalized non-gated child answer shape."""
    if T_table is None or len(p_child) == 0:
        return p_child
    sent = p_child.shape[1] - 1
    if sent < 1:
        return p_child
    p_asked = np.clip(p_parent @ T_table, 1e-6, 1.0 - 1e-6)
    shape = p_child[:, :sent].astype(np.float64)
    s = shape.sum(axis=1, keepdims=True)
    shape = np.where(s > 1e-9, shape / np.maximum(s, 1e-30), 1.0 / sent)
    out = np.empty_like(p_child, dtype=np.float64)
    out[:, :sent] = p_asked[:, None] * shape
    out[:, sent] = 1.0 - p_asked
    return out


# ----------------------------------------------------------- tree runners

def run_lgb(X, fit_pos, y_fit, pred_pos, K, n_est, ev_pos=None, y_ev=None):
    kw = {}
    if ev_pos is not None and len(ev_pos):
        kw = dict(eval_set=[(X.iloc[ev_pos], y_ev)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    m = lgb.LGBMClassifier(n_estimators=n_est, **PARAMS)
    m.fit(X.iloc[fit_pos], y_fit, **kw)
    p = np.zeros((len(pred_pos), K))
    if len(pred_pos):
        p[:, m.classes_] = m.predict_proba(X.iloc[pred_pos])
    return p, (m.best_iteration_ or n_est)


def run_cat(Xc, cat_cols, fit_pos, y_fit, pred_pos, K, n_est,
            ev_pos=None, y_ev=None):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(loss_function="MultiClass", iterations=n_est,
                           learning_rate=CAT_LR, depth=CAT_DEPTH,
                           l2_leaf_reg=5.0, thread_count=CAT_THREADS,
                           random_seed=SEED, verbose=0, task_type="GPU", devices="0",
                           allow_writing_files=False)
    kw = dict(cat_features=cat_cols)
    if ev_pos is not None and len(ev_pos):
        kw.update(eval_set=(Xc.iloc[ev_pos], y_ev),
                  early_stopping_rounds=30, use_best_model=True)
    m.fit(Xc.iloc[fit_pos], y_fit, **kw)
    p = np.zeros((len(pred_pos), K))
    if len(pred_pos):
        p[:, np.asarray(m.classes_, dtype=int)] = m.predict_proba(Xc.iloc[pred_pos])
    bi = m.get_best_iteration()
    return p, (n_est if bi is None else bi + 1)


# ----------------------------------------------------------- main predictors

def predict_lgb_exact(frame, schema):
    """Upgraded LightGBM + Multi-Task MLP + Two-Layer Gating for wide schemas (UNHCR)."""
    t0 = time.time()
    items = schema["items"]
    gated = schema["gated_value"]

    def options(name):
        vals = list(items[name]["values"])
        if items[name].get("gate") is not None:
            vals.append(gated)
        return vals

    given = [k for k, v in items.items() if v["class"] == "GIVEN"]
    targets = [k for k, v in items.items() if v["class"] == "PREDICT"]
    t_idx = {t: j for j, t in enumerate(targets)}

    X = pd.DataFrame(index=frame.index)
    for c in given:
        opts_c = options(c)
        cat = pd.Categorical(frame[c], categories=opts_c)
        if len(opts_c) <= MAX_CAT:
            X[c] = cat
        else:
            codes = cat.codes.astype(float)
            codes[codes < 0] = np.nan
            X[c] = codes

    n = len(frame)
    opts_of = {t: options(t) for t in targets}
    k_sizes = [len(opts_of[t]) for t in targets]
    Y = {t: pd.Categorical(frame[t], categories=opts_of[t]).codes.astype(int)
         for t in targets}
    Y_mat = np.column_stack([Y[t] for t in targets])

    complete = frame[targets].notna().all(axis=1).to_numpy()
    comp_pos = np.flatnonzero(complete)
    held_any = np.flatnonzero(~complete)
    rng = np.random.default_rng(SEED)
    rng.shuffle(comp_pos)
    n_val = int(len(comp_pos) * VAL_FRAC)
    va_pos, tr_pos = comp_pos[:n_val], comp_pos[n_val:]

    # 1. Multi-Task MLP (Pass 1: fit on tr_pos, validate on va_pos)
    mlp_va, mlp_held_fallback, mlp_ep = None, None, 20
    X_oh = None
    if USE_MLP and len(va_pos) >= 50 and time.time() - t0 < TIME_LIMIT:
        X_oh = build_onehot_given(frame, given, options)
        mlp_va, mlp_held_fallback, mlp_ep = fit_predict_mlp(
            X_oh, Y_mat, k_sizes, tr_pos, va_pos, held_any)

    # 2. LightGBM (Pass 1: fit on tr_pos, validate on va_pos)
    gate_info = {}
    store = {}
    for j, t in enumerate(targets):
        K = k_sizes[j]
        y_tr, y_va = Y[t][tr_pos], Y[t][va_pos]
        counts = np.bincount(y_tr[y_tr >= 0], minlength=K) + 1.0
        marg = counts / counts.sum()
        p_lgb_va = np.tile(marg, (len(va_pos), 1))
        best_n = None
        if (len(va_pos) and len(np.unique(y_tr)) > 1
                and time.time() - t0 < TIME_LIMIT):
            seen = np.isin(y_va, np.unique(y_tr))
            fit_kw = {}
            if seen.any():
                fit_kw = dict(
                    eval_set=[(X.iloc[va_pos[seen]], y_va[seen])],
                    callbacks=[lgb.early_stopping(30, verbose=False)])
            m = lgb.LGBMClassifier(n_estimators=600, **PARAMS)
            m.fit(X.iloc[tr_pos], y_tr, **fit_kw)
            p_lgb_va = np.zeros((len(va_pos), K))
            p_lgb_va[:, m.classes_] = m.predict_proba(X.iloc[va_pos])
            best_n = m.best_iteration_ or m.n_estimators

        gate_info[t] = build_given_gate_mask(frame, schema, t, opts_of[t], tr_pos, Y[t])
        p_m_va = mlp_va[:, j, :K] if mlp_va is not None else None
        store[t] = dict(y_va=y_va, marg=marg, p_lgb=p_lgb_va, p_mlp=p_m_va, best_n=best_n)

    # 3. Choose optimal global MLP blend weight on validation
    best_mw = 0.0
    if mlp_va is not None and len(va_pos) >= 50:
        sc_by_w = {}
        for mw in MLP_W_GRID:
            tot, cnt = 0.0, 0
            for t, s in store.items():
                p = (1.0 - mw) * s["p_lgb"] + mw * s["p_mlp"]
                p = apply_given_gate(p, va_pos, gate_info[t])
                tot += np.log(np.clip(p[np.arange(len(s["y_va"])), s["y_va"]], 1e-12, 1.0)).sum()
                cnt += len(s["y_va"])
            sc_by_w[mw] = tot / max(cnt, 1)
        top = max(sc_by_w, key=sc_by_w.get)
        if sc_by_w[top] - sc_by_w[0.0] >= MLP_MIN_GAIN:
            best_mw = top

    # Combine LGB + MLP and apply GIVEN gates on validation
    base_va = {}
    for t, s in store.items():
        p = ((1.0 - best_mw) * s["p_lgb"] + best_mw * s["p_mlp"]
             if s["p_mlp"] is not None else s["p_lgb"])
        base_va[t] = apply_given_gate(p, va_pos, gate_info[t])

    # 4. Skip-aware PREDICT-parent propagation (A_data) validation check
    pred_gate_w = {}
    if USE_PREDICT_GATE and len(va_pos) >= 50:
        for t in targets:
            gate = items[t].get("gate")
            par = gate.get("parent") if gate else None
            if par not in t_idx or gate_info[t] is not None:
                continue
            T_tab = fit_parent_transition(Y[t], Y[par], tr_pos, len(opts_of[t]), len(opts_of[par]))
            if T_tab is None:
                continue
            p_ad = apply_parent_transition(base_va[t], base_va[par], T_tab)
            best_gw, best_l = 0.0, nll_1d(base_va[t], store[t]["y_va"])
            for gw in (0.5, 1.0):
                p_mix = (1.0 - gw) * base_va[t] + gw * p_ad
                l = nll_1d(p_mix, store[t]["y_va"])
                if l < best_l - 1e-4:
                    best_l, best_gw = l, gw
            if best_gw > 0.0:
                pred_gate_w[t] = (par, best_gw)
                base_va[t] = (1.0 - best_gw) * base_va[t] + best_gw * p_ad

    # 5. Global (W, T) calibration search
    def pooled(W, T):
        total, cells = 0.0, 0
        for t, s in store.items():
            q = blend(base_va[t], s["marg"], W, T)
            q = apply_given_gate(q, va_pos, gate_info[t])
            total += np.log(np.clip(q[np.arange(len(s["y_va"])), s["y_va"]], 1e-12, 1.0)).sum()
            cells += len(s["y_va"])
        return total / max(cells, 1)

    best_W, best_T = 0.3, 1.0
    if len(va_pos) >= 50:
        best_score = -np.inf
        for T in T_GRID:
            for W in W_GRID:
                s = pooled(W, T)
                if s > best_score:
                    best_score, best_W, best_T = s, W, T

    # 6. Pass 2: Refit MLP on all complete rows
    mlp_held = mlp_held_fallback
    if (USE_MLP and best_mw > 0.0 and len(held_any) and X_oh is not None
            and time.time() - t0 < TIME_LIMIT):
        _, mlp_held, _ = fit_predict_mlp(
            X_oh, Y_mat, k_sizes, comp_pos, np.zeros(0, dtype=int), held_any,
            epochs=max(10, int(mlp_ep * 1.1)))

    # 7. Pass 2: Refit LightGBM on all observed rows, predict on held_any
    raw_held = {}
    marg_full = {}
    for j, t in enumerate(targets):
        K = k_sizes[j]
        fit_pos = np.flatnonzero(frame[t].notna().to_numpy())
        y_fit = Y[t][fit_pos]
        counts = np.bincount(y_fit, minlength=K) + 1.0
        marg = counts / counts.sum()
        marg_full[t] = marg

        p_lgb = np.tile(marg, (len(held_any), 1))
        bn = store[t]["best_n"]
        if (bn is not None and len(held_any) and len(np.unique(y_fit)) > 1
                and time.time() - t0 < TIME_LIMIT):
            m = lgb.LGBMClassifier(n_estimators=max(20, int(bn * 1.2)), **PARAMS)
            m.fit(X.iloc[fit_pos], y_fit)
            p_lgb = np.zeros((len(held_any), K))
            p_lgb[:, m.classes_] = m.predict_proba(X.iloc[held_any])

        if mlp_held is not None and best_mw > 0.0:
            p = (1.0 - best_mw) * p_lgb + best_mw * mlp_held[:, j, :K]
        else:
            p = p_lgb
        raw_held[t] = apply_given_gate(p, held_any, gate_info[t])

    # 8. Apply PREDICT-parent gate propagation on held_any
    if pred_gate_w and len(held_any):
        for t, (par, gw) in pred_gate_w.items():
            both_obs = np.flatnonzero(frame[t].notna().to_numpy() & frame[par].notna().to_numpy())
            T_tab = fit_parent_transition(Y[t], Y[par], both_obs, len(opts_of[t]), len(opts_of[par]))
            if T_tab is not None:
                p_par = raw_held[par].copy()
                # If a row in held_any actually has the parent observed, use its true one-hot
                obs_in_held = Y[par][held_any] >= 0
                if obs_in_held.any():
                    p_par[obs_in_held] = 0.0
                    p_par[obs_in_held, Y[par][held_any][obs_in_held]] = 1.0
                p_ad = apply_parent_transition(raw_held[t], p_par, T_tab)
                raw_held[t] = (1.0 - gw) * raw_held[t] + gw * p_ad

    # 9. Final calibration blend, GIVEN-gate enforcement, and formatting
    held_map = {int(pos): i for i, pos in enumerate(held_any)}
    probs = {}
    for t in targets:
        out = blend(raw_held[t], marg_full[t], best_W, best_T)
        out = apply_given_gate(out, held_any, gate_info[t])
        out = np.clip(out, 1e-9, None)
        out = out / out.sum(axis=1, keepdims=True)
        probs[t] = out

    result = []
    vals_mat = frame[targets].isna().to_numpy()
    for pos in range(n):
        hi = held_map.get(pos)
        if hi is None:
            continue
        for j, t in enumerate(targets):
            if vals_mat[pos, j]:
                result.append(probs[t][hi].tolist())
    return result


def predict_auto_cat_lgb(frame, schema):
    """Upgraded CatBoost + LightGBM + Multi-Task MLP + Gating for UNICEF & World Bank."""
    t0 = time.time()
    items = schema["items"]
    gated = schema["gated_value"]

    def options(name):
        vals = list(items[name]["values"])
        if items[name].get("gate"):
            vals.append(gated)
        return vals

    given = [k for k, v in items.items() if v["class"] == "GIVEN"]
    targets = [k for k, v in items.items() if v["class"] == "PREDICT"]
    t_idx = {t: j for j, t in enumerate(targets)}

    w_grid_use = np.r_[W_GRID, 0.98, 0.995] if len(targets) <= 20 else W_GRID

    X = pd.DataFrame(index=frame.index)
    Xc = pd.DataFrame(index=frame.index)
    cat_cols = []
    for c in given:
        opts_c = options(c)
        cat = pd.Categorical(frame[c], categories=opts_c)
        num = cat.codes.astype(float)
        num[num < 0] = np.nan
        X[c] = cat if len(opts_c) <= MAX_CAT else num
        Xc[c] = num
        Xc[c + "__cat"] = cat.codes.astype(str)
        cat_cols.append(c + "__cat")

    n = len(frame)
    opts_of = {t: options(t) for t in targets}
    k_sizes = [len(opts_of[t]) for t in targets]
    Y = {t: pd.Categorical(frame[t], categories=opts_of[t]).codes.astype(int)
         for t in targets}
    Y_mat = np.column_stack([Y[t] for t in targets])

    complete = frame[targets].notna().all(axis=1).to_numpy()
    comp_pos = np.flatnonzero(complete)
    held_any = np.flatnonzero(~complete)
    rng = np.random.default_rng(SEED)
    rng.shuffle(comp_pos)
    n_val = int(len(comp_pos) * VAL_FRAC)
    va_pos, tr_pos = comp_pos[:n_val], comp_pos[n_val:]

    # 1. Multi-Task MLP (Pass 1)
    mlp_va, mlp_held_fallback, mlp_ep = None, None, 20
    X_oh = None
    if USE_MLP and len(va_pos) >= 50 and time.time() - t0 < TIME_LIMIT:
        X_oh = build_onehot_given(frame, given, options)
        mlp_va, mlp_held_fallback, mlp_ep = fit_predict_mlp(
            X_oh, Y_mat, k_sizes, tr_pos, va_pos, held_any)

    # 2. LightGBM + CatBoost (Pass 1)
    gate_info = {}
    store = {}
    for j, t in enumerate(targets):
        K = k_sizes[j]
        y_tr, y_va = Y[t][tr_pos], Y[t][va_pos]
        counts = np.bincount(y_tr[y_tr >= 0], minlength=K) + 1.0
        marg = counts / counts.sum()
        s = dict(y_va=y_va, marg=marg, p_lgb=None, p_cat=None,
                 p_mlp=(mlp_va[:, j, :K] if mlp_va is not None else None),
                 n_lgb=None, n_cat=None)
        if len(va_pos) and len(np.unique(y_tr)) > 1:
            seen = np.isin(y_va, np.unique(y_tr))
            ev_pos = va_pos[seen] if seen.any() else None
            y_ev = y_va[seen] if seen.any() else None
            if time.time() - t0 < TIME_LIMIT:
                s["p_lgb"], s["n_lgb"] = run_lgb(
                    X, tr_pos, y_tr, va_pos, K, 600, ev_pos, y_ev)
            if time.time() - t0 < TIME_LIMIT:
                s["p_cat"], s["n_cat"] = run_cat(
                    Xc, cat_cols, tr_pos, y_tr, va_pos, K, CAT_ITERS,
                    ev_pos, y_ev)
        gate_info[t] = build_given_gate_mask(frame, schema, t, opts_of[t], tr_pos, Y[t])
        store[t] = s

    def item_p(s, mode):
        pl, pc, pm = s["p_lgb"], s["p_cat"], s["p_mlp"]
        if mode == "lgb":
            p = pl if pl is not None else pc
        elif mode == "cat":
            p = pc if pc is not None else pl
        elif mode == "avg":
            p = 0.5 * (pl + pc) if (pl is not None and pc is not None) else (pl if pl is not None else pc)
        elif mode == "lgb_mlp":
            p = 0.65 * pl + 0.35 * pm if (pl is not None and pm is not None) else pl
        elif mode == "cat_mlp":
            p = 0.65 * pc + 0.35 * pm if (pc is not None and pm is not None) else pc
        elif mode == "all":
            avail = [x for x in (pl, pc, pm) if x is not None]
            p = sum(avail) / len(avail) if avail else None
        else:
            p = pl
        if p is None:
            p = np.tile(s["marg"], (len(s["y_va"]), 1))
        return p

    def pooled(W, T, mode):
        total, cells = 0.0, 0
        for t, s in store.items():
            q = blend(item_p(s, mode), s["marg"], W, T)
            q = apply_given_gate(q, va_pos, gate_info[t])
            total += np.log(np.clip(q[np.arange(len(s["y_va"])), s["y_va"]], 1e-12, 1.0)).sum()
            cells += len(s["y_va"])
        return total / max(cells, 1)

    def search(mode):
        best = (-np.inf, 0.3, 1.0)
        for T in T_GRID:
            for W in w_grid_use:
                sc = pooled(W, T, mode)
                if sc > best[0]:
                    best = (sc, W, T)
        return best

    modes = ["lgb", "cat", "avg"]
    if mlp_va is not None:
        modes += ["lgb_mlp", "cat_mlp", "all"]

    if len(va_pos) >= 50:
        results = {m: search(m) for m in modes}
        chosen = max(results, key=lambda m: results[m][0])
        MLP_MODES = ("lgb_mlp", "cat_mlp", "all")
        if chosen in MLP_MODES:
            non_mlp = max(("lgb", "cat", "avg"), key=lambda m: results[m][0])
            if results[chosen][0] - results[non_mlp][0] < MLP_MIN_GAIN:
                chosen = non_mlp
        _, best_W, best_T = results[chosen]
    else:
        chosen = "lgb"
        best_W, best_T = 0.3, 1.0

    # Check PREDICT-parent gates on validation for the chosen mode
    base_va = {t: apply_given_gate(item_p(store[t], chosen), va_pos, gate_info[t]) for t in targets}
    pred_gate_w = {}
    if USE_PREDICT_GATE and len(va_pos) >= 50:
        for t in targets:
            gate = items[t].get("gate")
            par = gate.get("parent") if gate else None
            if par not in t_idx or gate_info[t] is not None:
                continue
            T_tab = fit_parent_transition(Y[t], Y[par], tr_pos, len(opts_of[t]), len(opts_of[par]))
            if T_tab is None:
                continue
            p_ad = apply_parent_transition(base_va[t], base_va[par], T_tab)
            best_gw, best_l = 0.0, nll_1d(base_va[t], store[t]["y_va"])
            for gw in (0.5, 1.0):
                p_mix = (1.0 - gw) * base_va[t] + gw * p_ad
                l = nll_1d(p_mix, store[t]["y_va"])
                if l < best_l - 1e-4:
                    best_l, best_gw = l, gw
            if best_gw > 0.0:
                pred_gate_w[t] = (par, best_gw)
                base_va[t] = (1.0 - best_gw) * base_va[t] + best_gw * p_ad

    # Pass 2: Refit MLP if chosen mode uses MLP
    need_mlp = chosen in ("lgb_mlp", "cat_mlp", "all")
    mlp_held = mlp_held_fallback
    if (need_mlp and len(held_any) and X_oh is not None
            and time.time() - t0 < TIME_LIMIT):
        _, mlp_held, _ = fit_predict_mlp(
            X_oh, Y_mat, k_sizes, comp_pos, np.zeros(0, dtype=int), held_any,
            epochs=max(10, int(mlp_ep * 1.1)))

    # Pass 2: Refit trees on all observed rows, predict on held_any
    raw_held = {}
    marg_full = {}
    for j, t in enumerate(targets):
        K = k_sizes[j]
        fit_pos = np.flatnonzero(frame[t].notna().to_numpy())
        y_fit = Y[t][fit_pos]
        counts = np.bincount(y_fit, minlength=K) + 1.0
        marg = counts / counts.sum()
        marg_full[t] = marg
        s = store[t]

        pl, pc = None, None
        pm = mlp_held[:, j, :K] if (need_mlp and mlp_held is not None) else None
        if (len(held_any) and len(np.unique(y_fit)) > 1
                and time.time() - t0 < TIME_LIMIT):
            if chosen in ("lgb", "avg", "lgb_mlp", "all") and s["n_lgb"] is not None:
                pl, _ = run_lgb(X, fit_pos, y_fit, held_any, K,
                                max(20, int(s["n_lgb"] * 1.2)))
            if chosen in ("cat", "avg", "cat_mlp", "all") and s["n_cat"] is not None:
                pc, _ = run_cat(Xc, cat_cols, fit_pos, y_fit, held_any, K,
                                max(20, int(s["n_cat"] * 1.2)))

        if chosen == "lgb":
            p = pl
        elif chosen == "cat":
            p = pc
        elif chosen == "avg":
            avail = [x for x in (pl, pc) if x is not None]
            p = sum(avail) / len(avail) if avail else None
        elif chosen == "lgb_mlp":
            p = 0.65 * pl + 0.35 * pm if (pl is not None and pm is not None) else (pl if pl is not None else pm)
        elif chosen == "cat_mlp":
            p = 0.65 * pc + 0.35 * pm if (pc is not None and pm is not None) else (pc if pc is not None else pm)
        else:
            avail = [x for x in (pl, pc, pm) if x is not None]
            p = sum(avail) / len(avail) if avail else None

        if p is None:
            p = np.tile(marg, (len(held_any), 1))
        raw_held[t] = apply_given_gate(p, held_any, gate_info[t])

    # Apply PREDICT-parent gates on held_any
    if pred_gate_w and len(held_any):
        for t, (par, gw) in pred_gate_w.items():
            both_obs = np.flatnonzero(frame[t].notna().to_numpy() & frame[par].notna().to_numpy())
            T_tab = fit_parent_transition(Y[t], Y[par], both_obs, len(opts_of[t]), len(opts_of[par]))
            if T_tab is not None:
                p_par = raw_held[par].copy()
                obs_in_held = Y[par][held_any] >= 0
                if obs_in_held.any():
                    p_par[obs_in_held] = 0.0
                    p_par[obs_in_held, Y[par][held_any][obs_in_held]] = 1.0
                p_ad = apply_parent_transition(raw_held[t], p_par, T_tab)
                raw_held[t] = (1.0 - gw) * raw_held[t] + gw * p_ad

    # Final calibration blend, GIVEN-gate enforcement, and formatting
    held_map = {int(pos): i for i, pos in enumerate(held_any)}
    probs = {}
    for t in targets:
        out = blend(raw_held[t], marg_full[t], best_W, best_T)
        out = apply_given_gate(out, held_any, gate_info[t])
        out = np.clip(out, 1e-9, None)
        out = out / out.sum(axis=1, keepdims=True)
        probs[t] = out

    result = []
    vals_mat = frame[targets].isna().to_numpy()
    for pos in range(n):
        hi = held_map.get(pos)
        if hi is None:
            continue
        for j, t in enumerate(targets):
            if vals_mat[pos, j]:
                result.append(probs[t][hi].tolist())
    return result


def predict(frame, schema):
    targets = [k for k, v in schema["items"].items() if v["class"] == "PREDICT"]
    if len(targets) > 100:
        return predict_lgb_exact(frame, schema)
    return predict_auto_cat_lgb(frame, schema)
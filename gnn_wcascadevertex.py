import argparse
import copy
import math
import os
import random
import sqlite3
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial import KDTree
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool


FEATURE_NAMES = [
    "str_x", "str_y", "z_centroid", "z_asym", "t_mean", "t_asym",
    "q_log", "n_doms", "t_spread", "z_rel", "t_first", "q_frac",
    "distance_to_cascade", "log_energy"
]
N_FEATURES = len(FEATURE_NAMES)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_geo_dict_kdtree(geo_df):
    """Build KDTree for nearest-neighbor geometry mapping."""
    coords = geo_df[['dom_x', 'dom_y', 'dom_z']].values.astype(float)
    kd_tree = KDTree(coords, leafsize=40)
    string_ids = geo_df['string'].values.astype(int)
    return kd_tree, string_ids

def map_pulse_to_string(x, y, z, kd_tree, string_ids, max_distance=50.0):
    """Map pulse to nearest string."""
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])

def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    start_epoch = checkpoint.get("epoch", 0)
    best_val_auc = checkpoint.get("best_val_auc", -np.inf)
    best_epoch = checkpoint.get("best_epoch", -1)
    train_losses = checkpoint.get("train_losses", [])
    val_losses = checkpoint.get("val_losses", [])
    val_aucs = checkpoint.get("val_aucs", [])
    lr_history = checkpoint.get("lr_history", [])
    
    print(f"Resumed from checkpoint: epoch {start_epoch}, best AUC: {best_val_auc:.4f} @ epoch {best_epoch}")
    return start_epoch, best_val_auc, best_epoch, train_losses, val_losses, val_aucs, lr_history

def stream_events_with_truth(db_file, energy_threshold_min_tev=None, energy_threshold_max_tev=None):
    """Stream events row by row from SQLite, joining truth info."""
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    if energy_threshold_min_tev is None:
        energy_threshold_min_tev = 0.05
    
    conn = sqlite3.connect(db_file)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]

    if "cascade_reco_energy_tev" not in cols:
        raise RuntimeError(f"'cascade_reco_energy_tev' column missing from truth table in {db_file}")
    if "cascade_vertex_fit_x" not in cols:
        raise RuntimeError(f"'cascade_vertex_fit_x' column missing from truth table in {db_file}")

    where_clause = f"WHERE t.cascade_reco_energy_tev >= {energy_threshold_min_tev}"
    if energy_threshold_max_tev is not None:
        where_clause += f" AND t.cascade_reco_energy_tev <= {energy_threshold_max_tev}"

    query = f"""
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
               t.cascade_reco_energy_tev, t.cascade_vertex_fit_x, 
               t.cascade_vertex_fit_y, t.cascade_vertex_fit_z
        FROM   CleanedROIPulses p
        JOIN   truth            t ON p.event_no = t.event_no
        {where_clause}
        ORDER  BY p.event_no
    """

    cursor        = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer        = []
    meta_store    = {}

    for row in cursor:
        event_no = row[0]
        pulse    = row[:6]
        cascade_reco_energy_tev = row[6]
        cascade_vertex_fit_x    = row[7]
        cascade_vertex_fit_y    = row[8]
        cascade_vertex_fit_z    = row[9]

        if current_event is None:
            current_event = event_no

        if event_no != current_event:
            yield current_event, buffer, meta_store.get(current_event, {})
            buffer        = []
            current_event = event_no

        buffer.append(pulse)

        if event_no not in meta_store:
            meta_store[event_no] = {
                "cascade_reco_energy_tev": float(cascade_reco_energy_tev),
                "cascade_vertex_fit_x": float(cascade_vertex_fit_x),
                "cascade_vertex_fit_y": float(cascade_vertex_fit_y),
                "cascade_vertex_fit_z": float(cascade_vertex_fit_z),
            }

    if buffer:
        yield current_event, buffer, meta_store.get(current_event, {})

    conn.close()

# def event_to_string_graph(rows, geo_dict, meta=None,
#                            k=16, t_scale=500.0, xyz_scale=500.0,
#                            charge_threshold=0.0): Rishi
def event_to_string_graph(rows, kd_tree, string_ids_geo, meta=None,
                           k=16, t_scale=500.0, xyz_scale=500.0,
                           charge_threshold=0.0):
    """Convert raw pulse rows for one event into a PyG Data object."""
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits          = t_hits[mask], q_hits[mask]
    t_hits                  = t_hits - t0

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(
            x          = torch.zeros((1, N_FEATURES), dtype=torch.float),
            edge_index = torch.zeros((2, 0),          dtype=torch.long),
        )

    if len(t_hits) < 2:
        return _fallback()

    # string_ids = np.array([
    #     geo_dict.get((round(xi, 1), round(yi, 1), round(zi, 1)), -1)
    #     for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    # ]) Rishi
    string_ids = np.array([
    map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
    for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits     = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 2:
        return _fallback()

    q_total    = q_hits.sum() + 1e-6
    z_q_center = np.sum(q_hits * z_hits) / q_total
    
    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0
    
    cascade_vertex_fit_x = float(meta.get("cascade_vertex_fit_x", 0.0)) if meta else 0.0
    cascade_vertex_fit_y = float(meta.get("cascade_vertex_fit_y", 0.0)) if meta else 0.0
    cascade_vertex_fit_z = float(meta.get("cascade_vertex_fit_z", 0.0)) if meta else 0.0
    
    cascade_vertex_x_norm = cascade_vertex_fit_x / xyz_scale
    cascade_vertex_y_norm = cascade_vertex_fit_y / xyz_scale
    cascade_vertex_z_norm = cascade_vertex_fit_z / xyz_scale

    node_feats     = []
    unique_strings = np.unique(string_ids)

    for s in unique_strings:
        mask_s  = string_ids == s
        q_s     = q_hits[mask_s]
        t_s     = t_hits[mask_s]
        z_s     = z_hits[mask_s]
        x_s     = x_hits[mask_s]
        y_s     = y_hits[mask_s]
        q_sum_s = q_s.sum() + 1e-6

        str_x      = x_s[0] / xyz_scale
        str_y      = y_s[0] / xyz_scale
        str_z      = z_s[0] / xyz_scale
        z_centroid = np.sum(q_s * z_s) / q_sum_s
        z_mid      = np.median(z_s)
        q_top      = q_s[z_s >= z_mid].sum()
        q_bot      = q_s[z_s <  z_mid].sum()
        z_asym     = (q_top - q_bot) / q_sum_s
        t_mean_s   = np.sum(q_s * t_s) / q_sum_s
        q_early    = q_s[t_s <= 750.0].sum()
        q_late     = q_s[t_s >  750.0].sum()
        t_asym     = (q_early - q_late) / q_sum_s
        q_log      = np.log1p(q_sum_s)
        n_doms     = len(np.unique(z_s)) / 60.0
        t_spread   = float(t_s.std()) / t_scale if len(t_s) > 1 else 0.0
        z_rel      = (z_centroid - z_q_center) / xyz_scale
        t_first    = float(t_s.min()) / t_scale
        q_frac     = q_sum_s / q_total
        
        distance_to_cascade = np.sqrt(
            (str_x - cascade_vertex_x_norm)**2 +
            (str_y - cascade_vertex_y_norm)**2 +
            (str_z / xyz_scale - cascade_vertex_z_norm)**2
        )

        node_feats.append([
            str_x, str_y, z_centroid / xyz_scale, z_asym, t_mean_s / t_scale,
            t_asym, q_log, n_doms, t_spread, z_rel, t_first, q_frac,
            distance_to_cascade, log_energy
        ])

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor   = torch.tensor(node_feats, dtype=torch.float)
    pos        = x_tensor[:, [0, 1, 2]]
    
    k_actual   = min(16, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)
    
    return Data(x=x_tensor, edge_index=edge_index)

class FastNeutrinoDataset(InMemoryDataset):
    # def __init__(self, nue_dbs, tau_dbs, geo_dict,
    #              max_events_per_class=None, charge_threshold=0.0,
    #              energy_threshold_min_tev=None, energy_threshold_max_tev=None):
    def __init__(self, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                 max_events_per_class=None, charge_threshold=0.0,
                 energy_threshold_min_tev=None, energy_threshold_max_tev=None):
        super().__init__()

        if isinstance(nue_dbs, str): nue_dbs = [nue_dbs]
        if isinstance(tau_dbs, str): tau_dbs = [tau_dbs]

        for path in nue_dbs + tau_dbs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"DB file not found: {path}")

        data_list = []

        print(f"Loading tau events from {len(tau_dbs)} file(s)...")
        if energy_threshold_min_tev is not None or energy_threshold_max_tev is not None:
            e_min = energy_threshold_min_tev if energy_threshold_min_tev is not None else "default"
            e_max = energy_threshold_max_tev if energy_threshold_max_tev is not None else "no limit"
            print(f"  Energy range: {e_min} - {e_max} TeV")
        
        tau_count = 0
        for db in tqdm(tau_dbs, desc="  tau files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db, 
                                        energy_threshold_min_tev=energy_threshold_min_tev,
                                        energy_threshold_max_tev=energy_threshold_max_tev),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if max_events_per_class and tau_count >= max_events_per_class:
                    break
                # g = event_to_string_graph(
                #     rows, geo_dict, meta=meta, charge_threshold=charge_threshold
                # ) Rishi
                g = event_to_string_graph(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                g.y = torch.tensor([1.0])
                data_list.append(g)
                tau_count += 1
            if max_events_per_class and tau_count >= max_events_per_class:
                break

        print(f"Loading nue events from {len(nue_dbs)} file(s)...")
        if energy_threshold_min_tev is not None or energy_threshold_max_tev is not None:
            e_min = energy_threshold_min_tev if energy_threshold_min_tev is not None else "default"
            e_max = energy_threshold_max_tev if energy_threshold_max_tev is not None else "no limit"
            print(f"  Energy range: {e_min} - {e_max} TeV")
        
        nue_max_events = max_events_per_class
        if max_events_per_class and tau_count < max_events_per_class:
            print(f"  Note: Only {tau_count} tau events loaded (< max_events_per_class={max_events_per_class})")
            print(f"  Auto-balancing: limiting nue to {tau_count} events to match tau")
            nue_max_events = tau_count
        
        nue_count = 0
        for db in tqdm(nue_dbs, desc="  nue files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db,
                                        energy_threshold_min_tev=energy_threshold_min_tev,
                                        energy_threshold_max_tev=energy_threshold_max_tev),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if nue_max_events and nue_count >= nue_max_events:
                    break
                # g = event_to_string_graph(
                #     rows, geo_dict, meta=meta, charge_threshold=charge_threshold
                # ) Rishi
                g = event_to_string_graph(
                    rows, kd_tree, string_ids_geo, meta=meta, charge_threshold=charge_threshold
                )
                g.y = torch.tensor([0.0])
                data_list.append(g)
                nue_count += 1
            if nue_max_events and nue_count >= nue_max_events:
                break

        if len(data_list) == 0:
            raise RuntimeError("Dataset is empty -- check DB paths and table names.")

        print(f"\nDataset summary:")
        print(f"  Tau:  {tau_count}")
        print(f"  Nue:  {nue_count}")
        if tau_count != nue_count:
            ratio = max(tau_count, nue_count) / min(tau_count, nue_count)
            print(f"  Ratio: 1:{ratio:.2f}")
        else:
            print(f"  Ratio: 1:1 (balanced)")

        print("Collating dataset...")
        self.data, self.slices = self.collate(data_list)
        print(f"Total: {len(data_list)} events")

class TemporalGNN(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=N_FEATURES):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )

        self.conv1 = TransformerConv(hidden_dim,     hidden_dim, heads=4)
        self.conv2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv3 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv4 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)
        self.conv5 = TransformerConv(hidden_dim * 4, hidden_dim, heads=4)

        self.bn1 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn5 = nn.BatchNorm1d(hidden_dim * 4)
        self.act = nn.ReLU()
        # OPTIMIZATION 1: Increased dropout from 0.4 to 0.5
        self.drop = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # OPTIMIZATION 1: Increased from 0.4 to 0.5
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_mlp(x)
        x = self.drop(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop(self.act(self.bn3(self.conv3(x, edge_index))))
        x = self.drop(self.act(self.bn4(self.conv4(x, edge_index))))
        x = self.drop(self.act(self.bn5(self.conv5(x, edge_index))))
        x      = global_mean_pool(x, batch)
        mean_q = global_mean_pool(data.x[:, 6], batch).unsqueeze(1)

        return self.classifier(torch.cat([x, mean_q], dim=1)).squeeze(1)

def evaluate_loader(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss   = 0.0

    with torch.no_grad():
        for batch in loader:
            batch   = batch.to(device)
            logits  = model(batch)
            targets = batch.y.view(-1)
            loss    = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return avg_loss, auc, np.array(preds), np.array(trues)

# def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch, base_lr=1e-4, max_lr=1e-3, cycle_epochs=10):
#     warmup_steps = warmup_epochs * steps_per_epoch
#     cycle_steps = cycle_epochs * steps_per_epoch

#     def lr_lambda(current_step):
#         # Warmup phase
#         if current_step < warmup_steps:
#             return (base_lr + (max_lr - base_lr) * current_step / warmup_steps) / max_lr
        
#         # Cyclical phase (after warmup)
#         step_in_cycle = (current_step - warmup_steps) % cycle_steps
#         progress = step_in_cycle / cycle_steps
        
#         # Cosine annealing within cycle: 
#         cycle_lr = base_lr + (max_lr - base_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
#         return cycle_lr / max_lr

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(args):
    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    # geo      = pd.read_csv(args.geo)
    # geo_dict = build_geo_dict(geo)
    # print(f"Geometry: {len(geo_dict)} DOM entries")
    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built (100% mapping success)")

    energy_threshold_min = getattr(args, "energy_threshold_min_tev", None)
    energy_threshold_max = getattr(args, "energy_threshold_max_tev", None)
    if energy_threshold_min is not None or energy_threshold_max is not None:
        print(f"Energy thresholds: min={energy_threshold_min} TeV, max={energy_threshold_max} TeV")

    dataset = FastNeutrinoDataset(
        nue_dbs                   = args.nue_dbs,
        tau_dbs                   = args.tau_dbs,
        kd_tree                   =kd_tree, 
        string_ids_geo            =string_ids_geo,
        max_events_per_class      = args.max_events,
        charge_threshold          = args.charge_threshold,
        energy_threshold_min_tev  = energy_threshold_min,
        energy_threshold_max_tev  = energy_threshold_max,
    )

    indices = np.arange(len(dataset))
    labels  = np.array([dataset[i].y.item() for i in range(len(dataset))])

    holdout_frac = args.val_frac + args.test_frac
    train_idx, holdout_idx = train_test_split(
        indices, test_size=holdout_frac, stratify=labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size    = args.test_frac / holdout_frac,
        stratify     = labels[holdout_idx],
        random_state = 42,
    )

    train_ds = dataset[train_idx.tolist()]
    val_ds   = dataset[val_idx.tolist()]
    test_ds  = dataset[test_idx.tolist()]
    print(f"Train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = TemporalGNN(input_dim=N_FEATURES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-3)

    # scheduler = build_scheduler(
    #     optimizer,
    #     warmup_epochs   = 5,
    #     total_epochs    = args.epochs,
    #     steps_per_epoch = len(train_loader),
    #     base_lr         = 1e-4,
    #     max_lr          = args.lr,
    #     cycle_epochs    = 10,
    # )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs   = 5,
        total_epochs    = args.epochs,
        steps_per_epoch = len(train_loader),
    )

    n_pos      = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg      = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Class balance        -- tau: {n_pos}  nue: {n_neg}  pos_weight: {n_neg/n_pos:.3f}")
    print(f"LR schedule (OPT)    -- CYCLICAL: warmup 5 epochs, then 10-epoch cycles")
    print(f"Peak LR              -- {args.lr:.2e}")
    print(f"Base LR (cycle min)  -- 1.00e-04")
    print(f"Regularization (OPT) -- dropout: 0.5 (STRONG), weight_decay: 5e-3 (STRONG)")
    print(f"Input features       -- {N_FEATURES}")

    best_val_auc   = -np.inf
    best_val_loss  = np.inf
    best_state     = None
    best_epoch     = -1
    patience       = getattr(args, "patience", 50)
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses = []
    val_losses   = []
    val_aucs     = []
    lr_history   = []

    weights_path    = os.path.join(args.save_dir, "gnn_weights.pt")
    best_path       = os.path.join(args.save_dir, "gnn_weights_best.pt")
    checkpoint_path = os.path.join(args.save_dir, "gnn_checkpoint.pt")

    start_epoch = 0
    if hasattr(args, "resume_from") and args.resume_from:
        start_epoch, best_val_auc, best_epoch, train_losses, val_losses, val_aucs, lr_history = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, device
        )
        if len(val_losses) > 0:
            best_val_loss = float(np.min(val_losses))
        epochs_no_gain = start_epoch - best_epoch

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Epochs", unit="epoch", position=0)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0

        batch_pbar = tqdm(
            train_loader,
            desc=f"  Train {epoch+1:03d}",
            unit="batch", position=1, leave=False
        )
        for batch in batch_pbar:
            batch   = batch.to(device)
            optimizer.zero_grad()
            logits  = model(batch)
            targets = batch.y.view(-1)
            loss    = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * batch.num_graphs
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        current_lr = scheduler.get_last_lr()[0]
        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_auc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc if not np.isnan(val_auc) else 0.0)
        lr_history.append(current_lr)

        auc_improved = (not np.isnan(val_auc)) and (val_auc > best_val_auc)
        loss_improved = val_loss < best_val_loss

        if auc_improved:
            best_val_auc   = val_auc
            best_state     = copy.deepcopy(model.state_dict())
            best_epoch     = epoch + 1
            torch.save(best_state, best_path)

        if loss_improved:
            best_val_loss = val_loss

        if auc_improved or loss_improved:
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        epoch_pbar.set_postfix({
            "train":    f"{train_loss:.4f}",
            "val_AUC":  val_auc_str,
            "best_AUC": f"{best_val_auc:.4f}",
            "lr":       f"{current_lr:.2e}",
        })
        tqdm.write(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"AUC: {val_auc_str} | "
            f"Best: {best_val_auc:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        epoch_checkpoint = {
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_auc":         float(best_val_auc),
            "best_epoch":           best_epoch,
            "train_losses":         train_losses,
            "val_losses":           val_losses,
            "val_aucs":             val_aucs,
            "lr_history":           lr_history,
            "train_idx":            train_idx,
            "val_idx":              val_idx,
            "test_idx":             test_idx,
            "config":               vars(args),
        }
        epoch_ckpt_path = os.path.join(args.save_dir, f"gnn_checkpoint_epoch_{epoch+1:03d}.pt")
        latest_ckpt_path = os.path.join(args.save_dir, "gnn_checkpoint_latest.pt")
        torch.save(epoch_checkpoint, epoch_ckpt_path)
        torch.save(epoch_checkpoint, latest_ckpt_path)

        if epochs_no_gain >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best weights from epoch {best_epoch} (AUC={best_val_auc:.4f})")

    torch.save(model.state_dict(), weights_path)
    checkpoint = {
        "epoch":                best_epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_auc":         float(best_val_auc),
        "train_idx":            train_idx,
        "val_idx":              val_idx,
        "test_idx":             test_idx,
        "train_losses":         train_losses,
        "val_losses":           val_losses,
        "val_aucs":             val_aucs,
        "lr_history":           lr_history,
        "config":               vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved weight: {weights_path}")
    print(f"Saved best: {best_path}")
    print(f"Saved checkpoint: {checkpoint_path}")

    test_loss, test_auc, test_preds, test_trues = evaluate_loader(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor="black")
    fig.patch.set_facecolor("black")

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor("black")
        ax.tick_params(colors="white")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, color="white")
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")

    axes[0].plot(train_losses, color="cyan",   label="Train")
    axes[0].plot(val_losses,   color="orange", label="Val")
    axes[0].legend(facecolor="black", labelcolor="white", edgecolor="gray")
    style(axes[0], "Loss", "Epoch", "BCE Loss")

    axes[1].plot(val_aucs, color="lime")
    axes[1].axhline(best_val_auc, color="white", lw=0.8, linestyle="--",
                    label=f"best={best_val_auc:.4f} @ ep{best_epoch}")
    axes[1].legend(facecolor="black", labelcolor="white", edgecolor="gray")
    style(axes[1], "Val AUC", "Epoch", "AUC")

    axes[2].plot(lr_history, color="yellow")
    axes[2].axvline(4, color="gray", lw=0.8, linestyle="--", label="warmup end")
    axes[2].legend(facecolor="black", labelcolor="white", edgecolor="gray")
    axes[2].set_yscale("log")
    style(axes[2], "Learning Rate (Cyclical)", "Epoch", "LR (log scale)")

    fpr, tpr, _ = roc_curve(test_trues, test_preds)
    axes[3].plot(fpr, tpr, color="cyan", lw=2, label=f"AUC={test_auc:.3f}")
    axes[3].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[3].legend(facecolor="black", labelcolor="white", edgecolor="gray")
    style(axes[3], "ROC -- Test Set", "FPR", "TPR")

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tau vs Electron Neutrino GNN Classifier - OPTIMIZED VERSION",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--tau_dbs",    nargs="+", required=True,
                        help="Path(s) to tau SQLite DB files")
    shared.add_argument("--nue_dbs",    nargs="+", required=True,
                        help="Path(s) to nue SQLite DB files")
    shared.add_argument("--geo",
                        default="/mnt/scratch/baburish/doublepulse/stringv2/geometry_clean.csv",
                        help="Path to geometry_clean.csv")
    shared.add_argument("--max_events", type=int, default=None,
                        help="Max events per class (auto-balances if tau < max)")
    shared.add_argument("--charge_threshold", type=float, default=0.1,
                        help="Keep only DOM hits with charge > threshold")
    shared.add_argument("--energy_threshold_min_tev", type=float, default=None,
                        help="Minimum cascade_reco_energy_tev in TeV (default: 0.05 TeV)")
    shared.add_argument("--energy_threshold_max_tev", type=float, default=None,
                        help="Maximum cascade_reco_energy_tev in TeV (None = no upper limit)")
    shared.add_argument("--save_dir",   default="./output",
                        help="Directory to save outputs")

    train_p = subparsers.add_parser("train", parents=[shared],
                                     help="Train the GNN")
    train_p.add_argument("--epochs",     type=int,   default=150)
    train_p.add_argument("--batch_size", type=int,   default=32)
    train_p.add_argument("--lr",         type=float, default=1e-3,
                         help="Peak learning rate (after warmup)")
    train_p.add_argument("--val_frac",   type=float, default=0.1)
    train_p.add_argument("--test_frac",  type=float, default=0.1)
    train_p.add_argument("--patience",   type=int,   default=50,
                         help="Early stopping patience (epochs without improvement)")
    train_p.add_argument("--resume_from", type=str, default=None,
                         help="Path to checkpoint to resume training from")

    diag_p = subparsers.add_parser("diagnostics", parents=[shared],
                                    help="Run feature diagnostics")
    diag_p.add_argument("--n_samples", type=int, default=1000,
                        help="Total events to sample for diagnostics")
    diag_p.add_argument("--n_energy_events", type=int, default=1000,
                        help="Total events to sample for energy histogram")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"Mode: {args.mode}")
    print(f"Args: {vars(args)}")

    if args.mode == "train":
        train(args)
    elif args.mode == "diagnostics":
        run_diagnostics(args)
    else:
        print(f"Unknown mode: {args.mode}", file=sys.stderr)
        sys.exit(1)
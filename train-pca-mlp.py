"""
train_hybrid_full.py

Trains the actual HybridGNNClassifier (GNN + PCA fusion, as originally
defined) using SGD, with the same train/val split methodology as the
PCA-only and GNN-only baseline scripts, for direct comparison.
"""

import argparse
import copy
import glob
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from tqdm import tqdm


GNN_FEATURE_NAMES = [
    "str_x", "str_y", "z_centroid", "z_asym", "t_mean", "t_asym",
    "q_log", "n_doms", "t_spread", "z_rel", "t_first", "q_frac",
    "distance_to_cascade", "log_energy"
]
N_GNN_FEATURES = len(GNN_FEATURE_NAMES)

PCA_FEATURE_NAMES = [
    "pc1_var", "pc2_var", "pc3_var", "elongation",
    "depth_loading", "time_loading", "charge_loading",
    "n_strings", "n_doms", "n_hits", "total_charge",
    "log_energy", "pc1_pc2_sum",
]
N_PCA_FEATURES = len(PCA_FEATURE_NAMES)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_geo_dict_kdtree(geo_df):
    coords = geo_df[["dom_x", "dom_y", "dom_z"]].values.astype(float)
    kd_tree = KDTree(coords, leafsize=40)
    string_ids = geo_df["string"].values.astype(int)
    return kd_tree, string_ids


def map_pulse_to_string(x, y, z, kd_tree, string_ids, max_distance=50.0):
    distance, idx = kd_tree.query([x, y, z])
    if distance > max_distance:
        return -1
    return int(string_ids[idx])


def stream_events_with_truth(db_file):
    """Streams events joined with truth AND reco, so cascade vertex/energy
    meta is real per-event data, not a constant fallback."""
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    conn = sqlite3.connect(db_file)
    query = """
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
               r.cascade_vertex_fit_x, r.cascade_vertex_fit_y, r.cascade_vertex_fit_z,
               r.cascade_reco_energy_tev
        FROM   CleanedROIPulses p
        JOIN   truth t ON p.event_no = t.event_no
        JOIN   reco  r ON p.event_no = r.event_no
        ORDER  BY p.event_no
    """
    cursor = conn.cursor()
    cursor.execute(query)
    current_event = None
    buffer = []
    current_meta = {}

    for row in cursor:
        event_no = row[0]
        pulse = row[:6]
        meta = {
            "cascade_vertex_fit_x": row[6],
            "cascade_vertex_fit_y": row[7],
            "cascade_vertex_fit_z": row[8],
            "cascade_reco_energy_tev": row[9],
        }

        if current_event is None:
            current_event = event_no
            current_meta = meta

        if event_no != current_event:
            yield current_event, buffer, current_meta
            buffer = []
            current_event = event_no
            current_meta = meta

        buffer.append(pulse)

    if buffer:
        yield current_event, buffer, current_meta

    conn.close()


# --------------------------------------------------------------------------
# PCA feature computation
# --------------------------------------------------------------------------

def compute_pca_features(rows, kd_tree, string_ids_geo, meta=None,
                          t_scale=500.0, xyz_scale=500.0, charge_threshold=0.0):
    rows = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0 = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits = t_hits[mask], q_hits[mask]
    t_hits = t_hits - t0

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits = t_hits[q_mask], q_hits[q_mask]

    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 4 or len(np.unique(string_ids)) < 2:
        return {
            "is_valid": False,
            "pc1_var": 0.0, "pc2_var": 0.0, "pc3_var": 1.0,
            "elongation": 1.0, "depth_loading": 0.0, "time_loading": 0.0,
            "charge_loading": 0.0, "n_strings": 0.0, "n_doms": 0.0, "n_hits": 0.0,
            "total_charge": 0.0, "log_energy": 0.0, "pc1_pc2_sum": 0.0,
        }

    X_spatial = np.column_stack([x_hits, y_hits, z_hits])
    charges = q_hits
    weights = charges / charges.sum()
    X_centered = X_spatial - np.average(X_spatial, weights=weights, axis=0)
    cov_weighted = np.cov(X_centered.T, aweights=weights)
    eigenvals_spatial, _ = np.linalg.eigh(cov_weighted)
    idx = np.argsort(eigenvals_spatial)[::-1]
    eigenvals_spatial = eigenvals_spatial[idx]

    total_var = eigenvals_spatial.sum() + 1e-10
    pc1_var = float(eigenvals_spatial[0] / total_var)
    pc2_var = float(eigenvals_spatial[1] / total_var)
    pc3_var = float(eigenvals_spatial[2] / total_var)
    elongation = float(eigenvals_spatial[0] / (eigenvals_spatial[2] + 1e-10))

    X_temporal = np.column_stack([t_hits, q_hits, z_hits])
    X_temporal_scaled = StandardScaler().fit_transform(X_temporal)

    cov_temporal = np.cov(X_temporal_scaled.T)
    eigenvals_temporal, eigenvecs_temporal = np.linalg.eigh(cov_temporal)
    idx_t = np.argsort(eigenvals_temporal)[::-1]
    eigenvecs_temporal = eigenvecs_temporal[:, idx_t]

    pc1_loadings = eigenvecs_temporal[:, 0]
    depth_loading = float(np.abs(pc1_loadings[2]))
    time_loading = float(np.abs(pc1_loadings[0]))
    charge_loading = float(np.abs(pc1_loadings[1]))

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    n_strings = len(np.unique(string_ids))
    n_doms = len(np.unique(np.column_stack([x_hits, y_hits, z_hits]), axis=0))
    n_hits = len(t_hits)
    total_charge = float(charges.sum())

    return {
        "is_valid": True,
        "pc1_var": pc1_var, "pc2_var": pc2_var, "pc3_var": pc3_var,
        "elongation": min(elongation, 50.0),
        "depth_loading": depth_loading, "time_loading": time_loading,
        "charge_loading": charge_loading,
        "n_strings": float(n_strings), "n_doms": float(n_doms), "n_hits": float(n_hits),
        "total_charge": total_charge, "log_energy": log_energy,
        "pc1_pc2_sum": pc1_var + pc2_var,
    }


# --------------------------------------------------------------------------
# Graph construction (string-level, 14 features -- matches original)
# --------------------------------------------------------------------------

def event_to_string_graph(rows, kd_tree, string_ids_geo, meta=None,
                           k=16, t_scale=500.0, xyz_scale=500.0,
                           charge_threshold=0.0):
    rows = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    t0 = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits = t_hits[mask], q_hits[mask]
    t_hits = t_hits - t0

    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(x=torch.zeros((1, N_GNN_FEATURES), dtype=torch.float),
                    edge_index=torch.zeros((2, 0), dtype=torch.long))

    if len(t_hits) < 2:
        return _fallback()

    string_ids = np.array([
        map_pulse_to_string(xi, yi, zi, kd_tree, string_ids_geo)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 2:
        return _fallback()

    q_total = q_hits.sum() + 1e-6
    z_q_center = np.sum(q_hits * z_hits) / q_total

    cascade_reco_energy_tev = float(meta.get("cascade_reco_energy_tev", 0.05)) if meta else 0.05
    log_energy = np.log10(max(cascade_reco_energy_tev, 0.05)) / 6.0

    cascade_vertex_x_norm = (float(meta.get("cascade_vertex_fit_x", 0.0)) if meta else 0.0) / xyz_scale
    cascade_vertex_y_norm = (float(meta.get("cascade_vertex_fit_y", 0.0)) if meta else 0.0) / xyz_scale
    cascade_vertex_z_norm = (float(meta.get("cascade_vertex_fit_z", 0.0)) if meta else 0.0) / xyz_scale

    node_feats = []
    unique_strings = np.unique(string_ids)

    for s in unique_strings:
        mask_s = string_ids == s
        q_s = q_hits[mask_s]
        t_s = t_hits[mask_s]
        z_s = z_hits[mask_s]
        x_s = x_hits[mask_s]
        y_s = y_hits[mask_s]
        q_sum_s = q_s.sum() + 1e-6

        str_x = x_s[0] / xyz_scale
        str_y = y_s[0] / xyz_scale
        str_z = z_s[0] / xyz_scale
        z_centroid = np.sum(q_s * z_s) / q_sum_s
        z_mid = np.median(z_s)
        q_top = q_s[z_s >= z_mid].sum()
        q_bot = q_s[z_s < z_mid].sum()
        z_asym = (q_top - q_bot) / q_sum_s
        t_mean_s = np.sum(q_s * t_s) / q_sum_s
        q_early = q_s[t_s <= 750.0].sum()
        q_late = q_s[t_s > 750.0].sum()
        t_asym = (q_early - q_late) / q_sum_s
        q_log = np.log1p(q_sum_s)
        n_doms = len(np.unique(z_s)) / 60.0
        t_spread = float(t_s.std()) / t_scale if len(t_s) > 1 else 0.0
        z_rel = (z_centroid - z_q_center) / xyz_scale
        t_first = float(t_s.min()) / t_scale
        q_frac = q_sum_s / q_total

        distance_to_cascade = np.sqrt(
            (str_x - cascade_vertex_x_norm) ** 2 +
            (str_y - cascade_vertex_y_norm) ** 2 +
            (str_z - cascade_vertex_z_norm) ** 2
        )

        node_feats.append([str_x, str_y, z_centroid / xyz_scale, z_asym, t_mean_s / t_scale,
                            t_asym, q_log, n_doms, t_spread, z_rel, t_first, q_frac,
                            distance_to_cascade, log_energy])

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor = torch.tensor(node_feats, dtype=torch.float)
    pos = x_tensor[:, [0, 1, 2]]

    k_actual = min(k, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)

    return Data(x=x_tensor, edge_index=edge_index)


# --------------------------------------------------------------------------
# Dataset (mirrors HybridNeutrinoDataset from hybrid_common.py)
# --------------------------------------------------------------------------

class HybridNeutrinoDataset(InMemoryDataset):
    def __init__(self, nue_dbs, tau_dbs, kd_tree, string_ids_geo,
                 max_events_per_class=None, charge_threshold=0.0,
                 pca_scaler=None):
        super().__init__()

        if isinstance(nue_dbs, str): nue_dbs = [nue_dbs]
        if isinstance(tau_dbs, str): tau_dbs = [tau_dbs]

        for path in nue_dbs + tau_dbs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"DB file not found: {path}")

        data_list = []
        pca_features_list = []

        print(f"Loading tau events from {len(tau_dbs)} file(s)...")
        tau_count = 0
        for db in tqdm(tau_dbs, desc="  tau files", unit="file"):
            for eid, rows, meta in stream_events_with_truth(db):
                if max_events_per_class and tau_count >= max_events_per_class:
                    break

                g = event_to_string_graph(rows, kd_tree, string_ids_geo, meta=meta,
                                           charge_threshold=charge_threshold)
                g.y = torch.tensor([1.0])

                pca_feats = compute_pca_features(rows, kd_tree, string_ids_geo, meta=meta,
                                                  charge_threshold=charge_threshold)

                if pca_feats["is_valid"] and g.x.shape[0] >= 3:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    tau_count += 1

            if max_events_per_class and tau_count >= max_events_per_class:
                break
        print(f"  -> Loaded {tau_count} tau events")

        print(f"Loading nue events from {len(nue_dbs)} file(s)...")
        nue_max_events = max_events_per_class
        if max_events_per_class and tau_count < max_events_per_class:
            nue_max_events = tau_count

        nue_count = 0
        for db in tqdm(nue_dbs, desc="  nue files", unit="file"):
            for eid, rows, meta in stream_events_with_truth(db):
                if nue_max_events and nue_count >= nue_max_events:
                    break

                g = event_to_string_graph(rows, kd_tree, string_ids_geo, meta=meta,
                                           charge_threshold=charge_threshold)
                g.y = torch.tensor([0.0])

                pca_feats = compute_pca_features(rows, kd_tree, string_ids_geo, meta=meta,
                                                  charge_threshold=charge_threshold)

                if pca_feats["is_valid"] and g.x.shape[0] >= 3:
                    data_list.append(g)
                    pca_features_list.append(pca_feats)
                    nue_count += 1

            if nue_max_events and nue_count >= nue_max_events:
                break
        print(f"  -> Loaded {nue_count} nue events")

        if len(data_list) == 0:
            raise RuntimeError("Dataset is empty -- check DB paths and table names.")

        print(f"\nDataset summary:")
        print(f"  Tau: {tau_count}")
        print(f"  Nue: {nue_count}")

        self.data, self.slices = self.collate(data_list)

        self.pca_features_array = np.array([
            [feats[name] for name in PCA_FEATURE_NAMES]
            for feats in pca_features_list
        ], dtype=np.float32)

        if pca_scaler is None:
            self.pca_scaler = StandardScaler()
            self.pca_features_scaled = self.pca_scaler.fit_transform(self.pca_features_array)
        else:
            self.pca_scaler = pca_scaler
            self.pca_features_scaled = self.pca_scaler.transform(self.pca_features_array)

    def get_pca_features(self, idx):
        return torch.tensor(self.pca_features_scaled[idx], dtype=torch.float)


# --------------------------------------------------------------------------
# HybridGNNClassifier -- unmodified from hybrid_common.py
# --------------------------------------------------------------------------

class HybridGNNClassifier(nn.Module):
    def __init__(self, gnn_hidden_dim=64, pca_dim=13, fusion_dim=32):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, gnn_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gnn_hidden_dim),
        )

        self.conv1 = TransformerConv(gnn_hidden_dim, gnn_hidden_dim, heads=4)
        self.conv2 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)
        self.conv3 = TransformerConv(gnn_hidden_dim * 4, gnn_hidden_dim, heads=4)

        self.bn1 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(gnn_hidden_dim * 4)
        self.act = nn.ReLU()
        self.drop_gnn = nn.Dropout(0.4)

        gnn_output_dim = gnn_hidden_dim * 4 + 1

        self.pca_mlp = nn.Sequential(
            nn.Linear(pca_dim, fusion_dim),
            nn.ReLU(),
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(0.3),
        )
        pca_output_dim = fusion_dim

        self.fusion = nn.Sequential(
            nn.Linear(gnn_output_dim + pca_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, data, pca_features):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_mlp(x)
        x = self.drop_gnn(self.act(self.bn1(self.conv1(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn2(self.conv2(x, edge_index))))
        x = self.drop_gnn(self.act(self.bn3(self.conv3(x, edge_index))))
        gnn_output = global_mean_pool(x, batch)
        mean_q = global_mean_pool(data.x[:, 6], batch).unsqueeze(1)
        gnn_features = torch.cat([gnn_output, mean_q], dim=1)

        pca_output = self.pca_mlp(pca_features)

        fused = torch.cat([gnn_features, pca_output], dim=1)
        return self.fusion(fused).squeeze(-1)


class HybridDataLoader:
    """Wrapper to yield (batch, pca_features) tuples."""
    def __init__(self, gnn_loader, dataset, index_map):
        self.gnn_loader = gnn_loader
        self.dataset = dataset
        self.index_map = index_map  # maps position-in-subset -> original dataset index

    def __iter__(self):
        self.iter_gnn = iter(self.gnn_loader)
        self.pos = 0
        return self

    def __next__(self):
        batch = next(self.iter_gnn)
        orig_indices = self.index_map[self.pos: self.pos + batch.num_graphs]
        self.pos += batch.num_graphs
        pca_feats = torch.stack([self.dataset.get_pca_features(i) for i in orig_indices])
        return batch, pca_feats

    def __len__(self):
        return len(self.gnn_loader)


def evaluate_loader(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch, pca_feats in loader:
            batch = batch.to(device)
            pca_feats = pca_feats.to(device)
            logits = model(batch, pca_feats)
            targets = batch.y.view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item() * batch.num_graphs
            preds.extend(torch.sigmoid(logits).cpu().numpy())
            trues.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(trues, preds) if len(np.unique(trues)) >= 2 else float("nan")
    return avg_loss, auc, np.array(preds), np.array(trues)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    geo = pd.read_csv(args.geo)
    kd_tree, string_ids_geo = build_geo_dict_kdtree(geo)
    print(f"Geometry: {len(geo)} DOMs, KDTree built")

    tau_paths = sorted(glob.glob(os.path.join(args.tau_dir, args.tau_pattern)))[:args.n_files]
    nue_paths = sorted(glob.glob(os.path.join(args.nue_dir, args.nue_pattern)))[:args.n_files]

    dataset = HybridNeutrinoDataset(
        nue_dbs=nue_paths, tau_dbs=tau_paths,
        kd_tree=kd_tree, string_ids_geo=string_ids_geo,
        max_events_per_class=args.max_events, charge_threshold=args.charge_threshold,
    )

    indices = np.arange(len(dataset))
    labels = np.array([dataset[i].y.item() for i in range(len(dataset))])

    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_frac, stratify=labels, random_state=args.seed
    )

    train_ds = dataset[train_idx.tolist()]
    val_ds = dataset[val_idx.tolist()]
    print(f"Split data:  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader_gnn = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, num_workers=0)  # shuffle=False keeps index_map aligned
    val_loader_gnn = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    train_loader = HybridDataLoader(train_loader_gnn, dataset, train_idx)
    val_loader = HybridDataLoader(val_loader_gnn, dataset, val_idx)

    device = torch.device("cpu") if args.force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridGNNClassifier(gnn_hidden_dim=args.hidden_dim, pca_dim=N_PCA_FEATURES,
                                 fusion_dim=args.fusion_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    n_pos = max(int((labels[train_idx] == 1).sum()), 1)
    n_neg = max(int((labels[train_idx] == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -np.inf
    best_state = None
    best_epoch = -1
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses, val_losses, val_aucs = [], [], []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1:03d}", unit="batch", leave=False)
        for batch, pca_feats in pbar:
            batch = batch.to(device)
            pca_feats = pca_feats.to(device)

            optimizer.zero_grad()
            logits = model(batch, pca_feats)
            targets = batch.y.view(-1)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader.dataset)
        val_loss, val_auc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc if not np.isnan(val_auc) else 0.0)

        scheduler.step()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_gain = 0
        else:
            epochs_no_gain += 1

        val_auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "nan"
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
              f"| AUC: {val_auc_str} | Best: {best_val_auc:.4f}")

        if epochs_no_gain >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best epoch {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "hybrid_full_weights.pt"))
    print(f"\nBest val AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="black")
    ax.plot(train_losses, color="cyan", label="Train")
    ax.plot(val_losses, color="orange", label="Val")
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("Loss", color="white")
    ax.set_title("Full Hybrid (GNN+PCA) Training Curves", color="white")
    ax.legend(facecolor="black", labelcolor="white")
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "hybrid_full_curves.png"), dpi=150, facecolor="black")
    plt.close()

    print(f"Saved model and plots to {args.save_dir}")
    return best_val_auc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full HybridGNNClassifier (GNN+PCA fusion), unmodified architecture",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tau_dir", required=True)
    parser.add_argument("--nue_dir", required=True)
    parser.add_argument("--tau_pattern", default="nutau_gemini_ftp_65TeV_*.db")
    parser.add_argument("--nue_pattern", default="nue_gemini_ftp_65TeV_*.db")
    parser.add_argument("--geo", default="/path/to/geometry_clean.csv")
    parser.add_argument("--n_files", type=int, default=20)
    parser.add_argument("--max_events", type=int, default=None)
    parser.add_argument("--charge_threshold", type=float, default=0.1)
    parser.add_argument("--save_dir", default="./output_hybrid_full")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fusion_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
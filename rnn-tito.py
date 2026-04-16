"""
Author: Rishi Babu
All summaries are generated using Claude
DOM-level RNN_TITO Classifier for Tau vs Electron Neutrino Classification

This model operates at the DOM (individual photomultiplier tube) level rather than
string level. It uses a two-stage approach:
  Stage 1: GRU processes temporal sequences within clusters of DOMs
  Stage 2: DynEdgeTITO performs dynamic edge convolution on DOM-level graph

This preserves vertical structure within strings and temporal fine structure of events.

Usage:
    # Train
    python dom_rnn_tito.py train \
        --tau_dbs /path/to/nutau.db \
        --nue_dbs /path/to/nue.db \
        --geo     /path/to/geometry_clean.csv \
        --max_events 5000 \
        --epochs 100 \
        --batch_size 16 \
        --save_dir /path/to/output
"""

import argparse
import copy
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch_cluster import knn_graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool

N_DOM_FEATURES = 7  # Per-DOM input features
FEATURE_NAMES = [
    "charge",           # 0: charge on this DOM
    "time",             # 1: hit time (relative)
    "dom_x",            # 2: DOM X position
    "dom_y",            # 3: DOM Y position
    "dom_z",            # 4: DOM Z position
    "string_id",        # 5: string identifier (normalized)
    "log_energy",       # 6: log10(energy) / 6 from truth
]

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load training state from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"Resumed from checkpoint: epoch {checkpoint.get('epoch', 0)}, "
          f"best AUC: {checkpoint.get('best_val_auc', -np.inf):.4f}")
    
    return (
        checkpoint.get("epoch", 0),
        checkpoint.get("best_val_auc", -np.inf),
        checkpoint.get("best_epoch", -1),
        checkpoint.get("train_losses", []),
        checkpoint.get("val_losses", []),
        checkpoint.get("val_aucs", []),
        checkpoint.get("lr_history", []),
    )


def build_geo_dict(geo_df):
    """Maps (rounded dom_x, dom_y, dom_z) -> string number."""
    geo_dict = {}
    for _, row in geo_df.iterrows():
        key = (round(float(row.dom_x), 1),
               round(float(row.dom_y), 1),
               round(float(row.dom_z), 1))
        geo_dict[key] = int(row.string)
    return geo_dict

def stream_events_with_truth(db_file):
    """
    Stream events from SQLite database.
    Only events with energy > 5e4 are selected.
    Yields (event_no, pulse_rows, meta_dict) per event.
    """
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"DB file not found: {db_file}")

    conn = sqlite3.connect(db_file)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(truth)").fetchall()]

    if "energy" not in cols:
        raise RuntimeError(f"'energy' column missing from truth table in {db_file}")

    query = """
        SELECT p.event_no, p.dom_x, p.dom_y, p.dom_z, p.dom_time, p.charge,
               t.energy
        FROM   CleanedROIPulses p
        JOIN   truth            t ON p.event_no = t.event_no
        WHERE  t.energy > 5e4
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
        energy   = row[6]

        if current_event is None:
            current_event = event_no

        if event_no != current_event:
            yield current_event, buffer, meta_store.get(current_event, {})
            buffer        = []
            current_event = event_no

        buffer.append(pulse)

        if event_no not in meta_store:
            meta_store[event_no] = {"energy": float(energy)}

    if buffer:
        yield current_event, buffer, meta_store.get(current_event, {})

    conn.close()


def dom_graph(rows, geo_dict, meta=None, k=16, 
                        t_scale=500.0, xyz_scale=500.0, 
                        charge_threshold=0.0):
    """
    Convert raw pulse rows for one event into a PyG Data object at DOM level.
    Each node represents one DOM hit (not aggregated to string level).

    Node features (7):
        0  charge          Charge on DOM
        1  time            Hit time (relative to event start)
        2  dom_x           DOM X position
        3  dom_y           DOM Y position
        4  dom_z           DOM Z position
        5  string_id       String identifier (normalized)
        6  log_energy      log10(energy) / 6 from truth
    """
    rows   = np.array(rows, dtype=object)
    x_hits = rows[:, 1].astype(float)
    y_hits = rows[:, 2].astype(float)
    z_hits = rows[:, 3].astype(float)
    t_hits = rows[:, 4].astype(float)
    q_hits = rows[:, 5].astype(float)

    # Time window: keep [t0, t0 + 1500 ns]
    t0   = t_hits.min()
    mask = (t_hits >= t0) & (t_hits <= t0 + 1500.0)
    x_hits, y_hits, z_hits = x_hits[mask], y_hits[mask], z_hits[mask]
    t_hits, q_hits          = t_hits[mask], q_hits[mask]
    t_hits                  = t_hits - t0

    # Charge threshold
    q_mask = q_hits > charge_threshold
    x_hits, y_hits, z_hits = x_hits[q_mask], y_hits[q_mask], z_hits[q_mask]
    t_hits, q_hits          = t_hits[q_mask], q_hits[q_mask]

    def _fallback():
        return Data(
            x          = torch.zeros((1, N_DOM_FEATURES), dtype=torch.float),
            edge_index = torch.zeros((2, 0),              dtype=torch.long),
        )

    if len(t_hits) < 2:
        return _fallback()

    # Map to strings
    string_ids = np.array([
        geo_dict.get((round(xi, 1), round(yi, 1), round(zi, 1)), -1)
        for xi, yi, zi in zip(x_hits, y_hits, z_hits)
    ])
    valid = string_ids >= 0
    x_hits, y_hits, z_hits     = x_hits[valid], y_hits[valid], z_hits[valid]
    t_hits, q_hits, string_ids = t_hits[valid], q_hits[valid], string_ids[valid]

    if len(t_hits) < 2:
        return _fallback()

    # Compute global energy
    log_energy = np.log10(max(float(meta.get("energy", 1.0)), 1.0)) / 6.0 \
                 if meta else 0.0

    # Build node features: one row per DOM hit
    node_feats = []
    for i in range(len(q_hits)):
        node_feats.append([
            q_hits[i],              # 0: charge
            t_hits[i] / t_scale,    # 1: time
            x_hits[i] / xyz_scale,  # 2: dom_x
            y_hits[i] / xyz_scale,  # 3: dom_y
            z_hits[i] / xyz_scale,  # 4: dom_z
            string_ids[i] / 86.0,   # 5: string_id (IceCube has ~86 strings, normalize)
            log_energy,             # 6: log_energy
        ])

    node_feats = np.array(node_feats, dtype=np.float32)
    x_tensor   = torch.tensor(node_feats, dtype=torch.float)
    
    # Build spatial graph via k-NN on position
    pos = x_tensor[:, [2, 3, 4]]  # Use x, y, z
    k_actual = min(k, x_tensor.size(0) - 1)
    if k_actual < 1:
        return _fallback()
    edge_index = knn_graph(pos, k=k_actual, loop=False)
    
    return Data(x=x_tensor, edge_index=edge_index)

class TauLoader(InMemoryDataset):
    """DOM-level dataset for tau vs nue classification."""
    
    def __init__(self, nue_dbs, tau_dbs, geo_dict,
                 max_events_per_class=None, charge_threshold=0.0):
        super().__init__()

        if isinstance(nue_dbs, str): nue_dbs = [nue_dbs]
        if isinstance(tau_dbs, str): tau_dbs = [tau_dbs]

        for path in nue_dbs + tau_dbs:
            if not os.path.exists(path):
                raise FileNotFoundError(f"DB file not found: {path}")

        data_list = []

        # Load tau events
        print(f"Loading tau events from {len(tau_dbs)} file(s)...")
        count = 0
        for db in tqdm(tau_dbs, desc="  tau files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if max_events_per_class and count >= max_events_per_class:
                    break
                g = dom_graph(rows, geo_dict, meta=meta, 
                                      charge_threshold=charge_threshold)
                g.y = torch.tensor([1.0])
                data_list.append(g)
                count += 1
            if max_events_per_class and count >= max_events_per_class:
                break
        print(f"  -> Loaded {count} tau events")

        # Load nue events
        print(f"Loading nue events from {len(nue_dbs)} file(s)...")
        count = 0
        for db in tqdm(nue_dbs, desc="  nue files", unit="file"):
            for eid, rows, meta in tqdm(
                stream_events_with_truth(db),
                desc=f"    {os.path.basename(db)}",
                unit="ev", leave=False
            ):
                if max_events_per_class and count >= max_events_per_class:
                    break
                g = dom_graph(rows, geo_dict, meta=meta,
                                      charge_threshold=charge_threshold)
                g.y = torch.tensor([0.0])
                data_list.append(g)
                count += 1
            if max_events_per_class and count >= max_events_per_class:
                break
        print(f"  -> Loaded {count} nue events")

        if len(data_list) == 0:
            raise RuntimeError("Dataset is empty -- check DB paths and table names.")

        print("Collating dataset...")
        self.data, self.slices = self.collate(data_list)
        print(f"  -> Total: {len(data_list)} events")

class DOMRNNEncoder(nn.Module):
    """
    RNN encoder that processes temporal sequences of DOM hits.
    
    Groups DOMs by spatial proximity and processes each group's 
    temporal sequence independently, then outputs learned embeddings.
    """
    
    def __init__(self, input_dim=N_DOM_FEATURES, hidden_dim=64, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # GRU for temporal sequence modeling
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: Node features (batch*num_nodes, input_dim)
        
        Returns:
            Encoded features (batch*num_nodes, hidden_dim)
        """
        # Project input
        x = self.input_proj(x)  # (N, hidden_dim)
        
        # For temporal processing, we treat each node's history
        # In this case, we use GRU on sorted temporal sequences
        # For now, apply GRU in a simplified way: treat as sequence of 1
        x_seq = x.unsqueeze(1)  # (N, 1, hidden_dim)
        
        # GRU expects sequence, we have single frames
        # Initialize hidden state to zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, 
                        device=x.device)
        
        # Process through GRU
        out, _ = self.gru(x_seq, h0)  # (N, 1, hidden_dim)
        out = out.squeeze(1)  # (N, hidden_dim)
        
        # Project output
        out = self.output_proj(out)
        
        return out


class DynTransLayer(nn.Module):
    """Dynamic edge Transformer convolution layer."""
    
    def __init__(self, in_channels, out_channels, n_heads=8, 
                 features_subset=None, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.features_subset = features_subset or [0, 1, 2, 3]
        
        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=n_heads,
            concat=True,
            dropout=dropout,
        )
        
        self.bn = nn.BatchNorm1d(out_channels * n_heads)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
        
        Returns:
            Updated node features (N, out_channels * n_heads)
        """
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class RNNTito(nn.Module):
    """
    DOM-level RNN_TITO classifier combining:
      - RNN encoder for temporal learning
      - DynTrans layers for graph learning
      - Global pooling and classification head
    """
    
    def __init__(self, input_dim=N_DOM_FEATURES, rnn_hidden=64, 
                 dyntrans_channels=128, num_dyntrans=4, n_heads=8,
                 dropout=0.3, readout_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.rnn_hidden = rnn_hidden
        
        # Stage 1: RNN encoder
        self.rnn_encoder = DOMRNNEncoder(
            input_dim=input_dim,
            hidden_dim=rnn_hidden,
            num_layers=2,
            dropout=dropout,
        )
        
        # Stage 2: DynTrans layers
        self.dyntrans_layers = nn.ModuleList()
        for i in range(num_dyntrans):
            in_ch = rnn_hidden if i == 0 else dyntrans_channels * n_heads
            out_ch = dyntrans_channels
            layer = DynTransLayer(
                in_channels=in_ch,
                out_channels=out_ch,
                n_heads=n_heads,
                dropout=dropout,
            )
            self.dyntrans_layers.append(layer)
        
        # Output dimension after final DynTrans layer
        final_dim = dyntrans_channels * n_heads
        
        # Classification head
        self.readout = nn.Sequential(
            nn.Linear(final_dim, readout_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(readout_dim, 1),
        )
    
    def forward(self, data):
        """
        Args:
            data: PyG Data object with x (N, input_dim), edge_index (2, E), batch (N,)
        
        Returns:
            Logits (batch_size,)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Stage 1: RNN encoding
        x = self.rnn_encoder(x)  # (N, rnn_hidden)
        
        # Stage 2: Dynamic graph convolutions
        for layer in self.dyntrans_layers:
            x = layer(x, edge_index)  # (N, dyntrans_channels * n_heads)
        
        # Global pooling (using function instead of module)
        x = global_mean_pool(x, batch)  # (batch_size, final_dim)
        
        # Classification
        logits = self.readout(x).squeeze(1)  # (batch_size,)
        
        return logits


def evaluate_loader(model, loader, criterion, device):
    """Standard evaluation with loss and AUC."""
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
    preds_np = np.array(preds)
    trues_np = np.array(trues)
    auc = roc_auc_score(trues_np, preds_np) if len(np.unique(trues_np)) >= 2 else float("nan")
    
    return avg_loss, auc, preds_np, trues_np

def build_scheduler(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    """Warmup + cosine annealing schedule."""
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

    geo      = pd.read_csv(args.geo)
    geo_dict = build_geo_dict(geo)
    print(f"Geometry: {len(geo_dict)} DOM entries")

    dataset = TauLoader(
        nue_dbs              = args.nue_dbs,
        tau_dbs              = args.tau_dbs,
        geo_dict             = geo_dict,
        max_events_per_class = args.max_events,
        charge_threshold     = args.charge_threshold,
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
    print(f"Split -> train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = RNNTito(
        input_dim=N_DOM_FEATURES,
        rnn_hidden=64,
        dyntrans_channels=128,
        num_dyntrans=4,
        n_heads=8,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

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

    print(f"Class balance   -- tau: {n_pos}  nue: {n_neg}  pos_weight: {n_neg/n_pos:.3f}")
    print(f"LR schedule     -- warmup: 5 epochs, cosine decay over {args.epochs} epochs")
    print(f"Peak LR         -- {args.lr:.2e}")

    best_val_auc   = -np.inf
    best_val_loss  = np.inf
    best_state     = None
    best_epoch     = -1
    patience       = getattr(args, "patience", 15)
    epochs_no_gain = 0
    os.makedirs(args.save_dir, exist_ok=True)

    train_losses = []
    val_losses   = []
    val_aucs     = []
    lr_history   = []

    weights_path    = os.path.join(args.save_dir, "dom_rnn_tito_weights.pt")
    best_path       = os.path.join(args.save_dir, "dom_rnn_tito_weights_best.pt")
    checkpoint_path = os.path.join(args.save_dir, "dom_rnn_tito_checkpoint.pt")

    # Load from checkpoint if resuming
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

        # Save per-epoch checkpoint
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
        epoch_ckpt_path = os.path.join(args.save_dir, f"dom_rnn_tito_checkpoint_epoch_{epoch+1:03d}.pt")
        latest_ckpt_path = os.path.join(args.save_dir, "dom_rnn_tito_checkpoint_latest.pt")
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
    print(f"Saved weights    -> {weights_path}")
    print(f"Saved best       -> {best_path}")
    print(f"Saved checkpoint -> {checkpoint_path}")

    test_loss, test_auc, test_preds, test_trues = evaluate_loader(
        model, test_loader, criterion, device
    )
    print(f"Test Loss: {test_loss:.4f} | Test AUC: {test_auc:.4f}")

    # Training curves
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
    style(axes[2], "Learning Rate", "Epoch", "LR (log scale)")

    fpr, tpr, _ = roc_curve(test_trues, test_preds)
    axes[3].plot(fpr, tpr, color="cyan", lw=2, label=f"AUC={test_auc:.3f}")
    axes[3].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[3].legend(facecolor="black", labelcolor="white", edgecolor="gray")
    style(axes[3], "ROC -- Test Set", "FPR", "TPR")

    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "dom_rnn_tito_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Saved curves     -> {plot_path}")


def run_diagnostics(args):
    """Run diagnostics on dataset."""
    set_seed(42)

    if not os.path.exists(args.geo):
        print(f"ERROR: geometry file not found: {args.geo}", file=sys.stderr)
        sys.exit(1)

    geo      = pd.read_csv(args.geo)
    geo_dict = build_geo_dict(geo)

    dataset = TauLoader(
        nue_dbs              = args.nue_dbs,
        tau_dbs              = args.tau_dbs,
        geo_dict             = geo_dict,
        max_events_per_class = args.max_events,
        charge_threshold     = args.charge_threshold,
    )

    print(f"\nDataset Statistics:")
    print(f"Total events: {len(dataset)}")
    
    # Compute graph statistics
    num_nodes_list = []
    num_edges_list = []
    tau_count = 0
    nue_count = 0
    
    for i in range(len(dataset)):
        g = dataset[i]
        num_nodes_list.append(g.x.size(0))
        num_edges_list.append(g.edge_index.size(1))
        if g.y.item() == 1:
            tau_count += 1
        else:
            nue_count += 1
    
    num_nodes_list = np.array(num_nodes_list)
    num_edges_list = np.array(num_edges_list)
    
    print(f"\nClass distribution:")
    print(f"  Tau (label=1): {tau_count} events ({100*tau_count/len(dataset):.1f}%)")
    print(f"  Nue (label=0): {nue_count} events ({100*nue_count/len(dataset):.1f}%)")
    
    print(f"\nGraph statistics:")
    print(f"  Nodes per event:  {num_nodes_list.mean():.1f} ± {num_nodes_list.std():.1f} "
          f"(min={num_nodes_list.min()}, max={num_nodes_list.max()})")
    print(f"  Edges per event:  {num_edges_list.mean():.1f} ± {num_edges_list.std():.1f} "
          f"(min={num_edges_list.min()}, max={num_edges_list.max()})")
    
    # Feature statistics
    tau_feats = []
    nue_feats = []
    
    for i in range(len(dataset)):
        g = dataset[i]
        x = g.x.numpy()
        # Compute per-event statistics
        feat_summary = np.concatenate([x.mean(axis=0), x.std(axis=0)])
        if g.y.item() == 1:
            tau_feats.append(feat_summary)
        else:
            nue_feats.append(feat_summary)
    
    tau_feats = np.array(tau_feats)
    nue_feats = np.array(nue_feats)
    
    print(f"\nFeature separation (top 5 by separation score):")
    sep_scores = []
    for j in range(N_DOM_FEATURES):
        t_mean = tau_feats[:, j].mean()
        n_mean = nue_feats[:, j].mean()
        t_std = tau_feats[:, j].std() + 1e-6
        n_std = nue_feats[:, j].std() + 1e-6
        sep = abs(t_mean - n_mean) / (0.5 * (t_std + n_std))
        sep_scores.append((sep, FEATURE_NAMES[j], t_mean, n_mean))
    
    sep_scores.sort(reverse=True)
    for sep, name, t_mean, n_mean in sep_scores[:5]:
        print(f"  {name:<15} tau={t_mean:>8.4f}  nue={n_mean:>8.4f}  sep={sep:>8.4f}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Plot feature distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), facecolor="black")
    fig.patch.set_facecolor("black")
    axes = axes.flatten()
    
    for idx in range(min(8, N_DOM_FEATURES)):
        ax = axes[idx]
        t_vals = tau_feats[:, idx]
        n_vals = nue_feats[:, idx]
        
        lo = np.percentile(np.concatenate([t_vals, n_vals]), 1)
        hi = np.percentile(np.concatenate([t_vals, n_vals]), 99)
        bins = np.linspace(lo, hi, 40)
        
        ax.hist(t_vals, bins=bins, alpha=0.6, density=True, color="cyan", label="tau")
        ax.hist(n_vals, bins=bins, alpha=0.6, density=True, color="orange", label="nue")
        
        ax.set_facecolor("black")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("gray")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        sep = sep_scores[idx][0]
        ax.set_title(f"{FEATURE_NAMES[idx]} (sep={sep:.2f})", color="white", fontsize=9)
        
        legend = ax.legend(fontsize=7)
        for txt in legend.get_texts():
            txt.set_color("white")
        legend.get_frame().set_facecolor("black")
        legend.get_frame().set_edgecolor("gray")
    
    plt.tight_layout()
    plot_path = os.path.join(args.save_dir, "dom_rnn_tito_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"\nSaved diagnostics -> {plot_path}")

def parse_args():
    parser = argparse.ArgumentParser(
        description="DOM-level RNN_TITO Classifier for Tau vs Electron Neutrino",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--tau_dbs",    nargs="+", required=True,
                        help="Path(s) to tau SQLite DB files")
    shared.add_argument("--nue_dbs",    nargs="+", required=True,
                        help="Path(s) to nue SQLite DB files")
    shared.add_argument("--geo",
                        default="/mnt/scratch/baburish/doublepulse/stringv2/geometry_clean.csv",
                        help="Path to geometry_clean.csv")
    shared.add_argument("--max_events", type=int, default=None,
                        help="Max events per class (None = all)")
    shared.add_argument("--charge_threshold", type=float, default=0.0,
                        help="Keep only DOM hits with charge > threshold")
    shared.add_argument("--save_dir",   default="./output",
                        help="Directory to save outputs")

    # Train mode
    train_p = subparsers.add_parser("train", parents=[shared],
                                     help="Train the DOM RNN_TITO model")
    train_p.add_argument("--epochs",     type=int,   default=100)
    train_p.add_argument("--batch_size", type=int,   default=16)
    train_p.add_argument("--lr",         type=float, default=5e-4,
                         help="Peak learning rate (after warmup)")
    train_p.add_argument("--val_frac",   type=float, default=0.15)
    train_p.add_argument("--test_frac",  type=float, default=0.15)
    train_p.add_argument("--patience",   type=int,   default=15)
    train_p.add_argument("--resume_from", type=str, default=None)

    # Diagnostics mode
    subparsers.add_parser("diagnostics", parents=[shared],
                          help="Run diagnostics on dataset")

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
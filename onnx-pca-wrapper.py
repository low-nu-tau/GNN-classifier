"""
Export the trained Hybrid GNN+PCA model to ONNX.

This exporter preserves the original gnn-pca architecture, including:
- TransformerConv graph layers
- edge_index graph connectivity
- batch vector for graph-level pooling
"""

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool


N_GNN_FEATURES = 14
N_PCA_FEATURES = 13


class HybridGNNClassifier(nn.Module):
    """Your original model - unchanged"""
    
    def __init__(self, gnn_hidden_dim=64, pca_dim=13, fusion_dim=32):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(N_GNN_FEATURES, gnn_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(gnn_hidden_dim),
        )

        self.conv1 = TransformerConv(gnn_hidden_dim,     gnn_hidden_dim, heads=4)
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


class HybridGNNONNX(nn.Module):
    """
    ONNX wrapper that preserves the original model behavior.

    Inputs are raw tensors so ONNX Runtime can feed the graph directly.
    """
    
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, node_features, edge_index, batch, pca_features):
        """
        Tensor-only forward signature for ONNX export.

        Args:
            node_features: [num_nodes, 14]
            edge_index: [2, num_edges] int64 COO edge list
            batch: [num_nodes] int64 graph-id per node
            pca_features: [batch_size, 13]

        Returns:
            logits: [batch_size]
        """
        data = Data(x=node_features, edge_index=edge_index, batch=batch)
        return self.model(data, pca_features)


def _build_dummy_edge_index(num_nodes, device):
    """Build a small bidirectional ring so each node has graph neighbors."""
    if num_nodes < 2:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    src = torch.arange(0, num_nodes, device=device, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)

    # Bidirectional edges: i->i+1 and i+1->i
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])],
        dim=0,
    )
    return edge_index


def load_and_export(model_path, output_path="hybrid_model.onnx", device="cpu", opset=14):
    """
    Load the trained model and export to ONNX with proper dimension handling.
    """
    print(f"\n{'='*70}")
    print("EXPORTING HYBRID GNN+PCA TO ONNX (ARCHITECTURE-PRESERVING)")
    print(f"{'='*70}")
    
    device_obj = torch.device(device)
    print(f"\n1. Loading model from {model_path}...")
    
    # Load the original model with trained weights
    model = HybridGNNClassifier(gnn_hidden_dim=64, pca_dim=N_PCA_FEATURES, fusion_dim=32)
    
    try:
        state_dict = torch.load(model_path, map_location=device_obj)
        model.load_state_dict(state_dict)
        print("   ✓ Model weights loaded")
    except Exception as e:
        print(f"   ✗ Failed to load weights: {e}")
        return False
    
    model = model.to(device_obj)
    model.eval()
    
    # Create ONNX-compatible wrapper
    print("\n2. Creating ONNX-compatible wrapper...")
    onnx_model = HybridGNNONNX(model)
    onnx_model = onnx_model.to(device_obj)
    onnx_model.eval()
    print("   ✓ Wrapper created")
    
    # Test the wrapper forward pass first
    print("\n3. Testing wrapper forward pass...")
    with torch.no_grad():
        num_nodes = 20
        test_node_feats = torch.randn(num_nodes, N_GNN_FEATURES, device=device_obj)
        test_edge_index = _build_dummy_edge_index(num_nodes, device_obj)
        test_batch = torch.zeros(num_nodes, dtype=torch.long, device=device_obj)
        test_pca_feats = torch.randn(1, N_PCA_FEATURES, device=device_obj)
        
        try:
            output = onnx_model(test_node_feats, test_edge_index, test_batch, test_pca_feats)
            print(f"   ✓ Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"   ✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Create dummy inputs for ONNX export
    print("\n4. Creating dummy inputs for ONNX export...")
    dummy_num_nodes = 20
    dummy_node_features = torch.randn(dummy_num_nodes, N_GNN_FEATURES, dtype=torch.float32, device=device_obj)
    dummy_edge_index = _build_dummy_edge_index(dummy_num_nodes, device_obj)
    dummy_batch = torch.zeros(dummy_num_nodes, dtype=torch.long, device=device_obj)
    dummy_pca_features = torch.randn(1, N_PCA_FEATURES, dtype=torch.float32, device=device_obj)
    print(f"   Node features: {dummy_node_features.shape}")
    print(f"   Edge index: {dummy_edge_index.shape}")
    print(f"   Batch: {dummy_batch.shape}")
    print(f"   PCA features: {dummy_pca_features.shape}")
    
    # Export to ONNX
    print(f"\n5. Exporting to ONNX (opset {opset})...")
    try:
        torch.onnx.export(
            onnx_model,
            (dummy_node_features, dummy_edge_index, dummy_batch, dummy_pca_features),
            output_path,
            input_names=["node_features", "edge_index", "batch", "pca_features"],
            output_names=["logits"],
            dynamic_axes={
                "node_features": {0: "num_nodes"},
                "edge_index": {1: "num_edges"},
                "batch": {0: "num_nodes"},
                "pca_features": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )
        print(f"   ✓ Successfully exported to {output_path}")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify ONNX model
    print("\n6. Verifying ONNX model...")
    try:
        import onnx
        onnx_model_loaded = onnx.load(output_path)
        onnx.checker.check_model(onnx_model_loaded)
        print("   ✓ ONNX model is valid!")
    except ImportError:
        print("   ⚠ onnx not available (pip install onnx)")
    except Exception as e:
        print(f"   ✗ Verification failed: {e}")
        return False
    
    # Test inference
    print("\n7. Testing ONNX inference...")
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        for i in range(3):
            num_nodes = np.random.randint(10, 30)
            test_node_feats = np.random.randn(num_nodes, N_GNN_FEATURES).astype(np.float32)
            test_edge_index_t = _build_dummy_edge_index(num_nodes, torch.device("cpu"))
            test_edge_index = test_edge_index_t.cpu().numpy().astype(np.int64)
            test_batch = np.zeros((num_nodes,), dtype=np.int64)
            test_pca_feats = np.random.randn(1, N_PCA_FEATURES).astype(np.float32)
            
            outputs = sess.run(
                None,
                {
                    "node_features": test_node_feats,
                    "edge_index": test_edge_index,
                    "batch": test_batch,
                    "pca_features": test_pca_feats
                }
            )
            
            logits = outputs[0][0]
            prob = 1.0 / (1.0 + np.exp(-logits))
            pred = "tau" if prob > 0.5 else "nue"
            
            print(f"   Test {i+1}: {num_nodes} nodes → {pred} (prob={prob:.4f})")
        
        print("   ✓ Inference tests passed!")
    except ImportError:
        print("   ⚠ onnxruntime not available (pip install onnxruntime)")
    except Exception as e:
        print(f"   ✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n{'='*70}")
    print(f"✓ SUCCESS! ONNX model exported to: {output_path}")
    print(f"{'='*70}\n")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export Hybrid GNN to ONNX (Fixed Dimension Handling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "model_path",
        help="Path to saved model weights (.pt file)"
    )
    parser.add_argument(
        "--output",
        default="hybrid_model.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for export"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    # Export
    success = load_and_export(
        args.model_path,
        args.output,
        device=args.device,
        opset=args.opset
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
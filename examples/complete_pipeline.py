"""
Complete End-to-End Example

Demonstrates:
1. Graph construction from transactions
2. Model training with real metrics
3. Inference scoring
4. Fraud pattern detection
5. Explainability analysis
"""

import sys
import torch
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("AEGIS GRAPH SENTINEL 2.0 - PRODUCTION HTGNN EXAMPLE")
    logger.info("=" * 80)
    
    # Step 1: Graph Construction
    logger.info("\n[STEP 1] Graph Construction from Transactions")
    logger.info("-" * 80)
    
    from src.data.graph_constructor import (
        TemporalGraphConstructor, Transaction,
        create_sample_transactions
    )
    
    # Initialize graph constructor
    constructor = TemporalGraphConstructor(
        time_window_hours=24,
        feature_dim=64,
        temporal_dim=16,
        temporal_decay_lambda=0.01,
    )
    
    # Generate sample transactions (in production: from real data source)
    transactions = create_sample_transactions()
    logger.info(f"Generated {len(transactions)} sample transactions")
    
    # Add to graph
    constructor.add_transactions(transactions)
    
    # Construct PyG graph
    graph = constructor.construct_pyg_graph()
    logger.info(
        f"✓ Graph constructed: {graph['num_nodes']} nodes, "
        f"{graph['num_edges']} edges"
    )
    
    # Step 2: Model Initialization
    logger.info("\n[STEP 2] Initialize HTGNN Model")
    logger.info("-" * 80)
    
    from src.models.risk_model import FraudDetectionModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = FraudDetectionModel(
        node_feature_dim=64,
        hidden_dim=128,
        output_dim=64,
        num_node_types=5,
        num_edge_types=5,
        num_layers=2,
        heads=4,
        dropout=0.2,
        temporal_dim=16,
    )
    model.to(device)
    
    logger.info(f"✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Step 3: Create Training Dataset
    logger.info("\n[STEP 3] Create Training Dataset")
    logger.info("-" * 80)
    
    from torch.utils.data import DataLoader
    from src.training.production_trainer import SimpleGraphDataset, collate_graphs
    
    # Generate multiple graph snapshots
    graphs = []
    labels = []
    
    for i in range(20):  # Small dataset for demo
        constructor_i = TemporalGraphConstructor(
            time_window_hours=24,
            feature_dim=64,
            temporal_dim=16,
        )
        
        # Generate transactions
        txns = create_sample_transactions()
        
        # Randomly label as fraud or normal
        is_fraud = i % 10 >= 7
        
        # Add to graph
        constructor_i.add_transactions(txns)
        graph_i = constructor_i.construct_pyg_graph()
        
        graphs.append(graph_i)
        labels.append(1 if is_fraud else 0)
    
    # Create dataset
    dataset = SimpleGraphDataset(graphs, labels)
    
    # Split: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    
    train_dataset = torch.utils.data.Subset(
        dataset,
        range(train_size),
    )
    val_dataset = torch.utils.data.Subset(
        dataset,
        range(train_size, train_size + val_size),
    )
    test_dataset = torch.utils.data.Subset(
        dataset,
        range(train_size + val_size, len(dataset)),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_graphs,
    )
    
    logger.info(f"✓ Dataset created:")
    logger.info(f"  - Train: {len(train_dataset)}")
    logger.info(f"  - Val:   {len(val_dataset)}")
    logger.info(f"  - Test:  {len(test_dataset)}")
    
    # Step 4: Train Model
    logger.info("\n[STEP 4] Train HTGNN Model")
    logger.info("-" * 80)
    
    from src.training.production_trainer import ProductionTrainer
    
    config = {
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 4,
        'num_epochs': 20,  # Short for demo
        'early_stopping_patience': 5,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'loss': {
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 2.0,
        },
    }
    
    trainer = ProductionTrainer(
        model=model,
        config=config,
        device=device,
        output_dir='models',
    )
    
    logger.info("Starting training...")
    results = trainer.train(train_loader, val_loader, test_loader)
    
    logger.info(f"\n✓ Training complete!")
    logger.info(f"  - Best epoch: {results['best_epoch']}")
    logger.info(f"  - Best F1:    {results['best_val_f1']:.4f}")
    
    if 'test_metrics' in results:
        test_metrics = results['test_metrics']
        logger.info(f"  - Test F1:    {test_metrics['f1']:.4f}")
        logger.info(f"  - Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"  - Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  - Test Recall:    {test_metrics['recall']:.4f}")
    
    # Step 5: Load Best Model
    logger.info("\n[STEP 5] Load Best Model Checkpoint")
    logger.info("-" * 80)
    
    trainer.load_best_model()
    logger.info("✓ Best model loaded for inference")
    
    # Step 6: Fraud Pattern Detection
    logger.info("\n[STEP 6] Detect Fraud Patterns")
    logger.info("-" * 80)
    
    from src.features.fraud_pattern_detector import FraudPatternDetector
    
    detector = FraudPatternDetector(
        min_chain_length=3,
        max_hours_between_transfers=24,
    )
    
    # Detect various patterns
    mule_rings = detector.detect_mule_rings(transactions, datetime.utcnow())
    fan_in_hubs = detector.detect_fan_in_hubs(transactions, threshold_incoming=5)
    fan_out_hubs = detector.detect_fan_out_hubs(transactions, threshold_outgoing=8)
    velocity_anomalies = detector.detect_velocity_anomalies(transactions)
    
    logger.info(f"✓ Fraud patterns detected:")
    logger.info(f"  - Mule rings:          {len(mule_rings)}")
    logger.info(f"  - Fan-in hubs:         {len(fan_in_hubs)}")
    logger.info(f"  - Fan-out hubs:        {len(fan_out_hubs)}")
    logger.info(f"  - Velocity anomalies:  {len(velocity_anomalies)}")
    
    if mule_rings:
        for ring in mule_rings[:2]:
            logger.info(f"  - Mule ring: {ring['ring_accounts']} (risk={ring['risk_score']:.2f})")
    
    # Step 7: Real-Time Inference
    logger.info("\n[STEP 7] Real-Time Fraud Scoring")
    logger.info("-" * 80)
    
    from src.inference.production_scorer import ProductionRiskScorer
    
    # Initialize risk scorer with trained model
    risk_scorer = ProductionRiskScorer(
        model=model,
        graph_constructor=constructor,
        device=device,
        model_version='2.0.0',
        enable_heuristic_fallback=True,
    )
    
    # Score a sample transaction
    sample_transaction = {
        'transaction_id': 'TXN_DEMO_001',
        'source_account': 'ACC_NORMAL_0',
        'target_account': 'ACC_NORMAL_1',
        'amount': 50000,
        'timestamp': datetime.utcnow() - timedelta(minutes=5),
        'mode': 'UPI',
        'source_device_id': 'DEV_0',
        'source_ip': 'IP_0',
    }
    
    fraud_score = risk_scorer.score_transaction(
        transaction=sample_transaction,
        reference_time=datetime.utcnow(),
        k_hops=2,
    )
    
    logger.info(f"\n✓ Transaction scored:")
    logger.info(f"  - Transaction ID:      {fraud_score.transaction_id}")
    logger.info(f"  - Risk Score:          {fraud_score.risk_score:.4f}")
    logger.info(f"  - Decision:            {fraud_score.decision}")
    logger.info(f"  - Confidence:          {fraud_score.confidence:.4f}")
    logger.info(f"  - Inference Time:      {fraud_score.inference_time_ms:.2f}ms")
    logger.info(f"  - Graph Size:          {fraud_score.graph_size} nodes")
    logger.info(f"  - Breakdown:")
    for component, score in fraud_score.breakdown.items():
        logger.info(f"      - {component}: {score:.4f}")
    
    logger.info(f"\n  Explanation:")
    for line in fraud_score.explanation.split('\n'):
        logger.info(f"    {line}")
    
    # Step 8: Batch Scoring
    logger.info("\n[STEP 8] Batch Scoring")
    logger.info("-" * 80)
    
    batch_transactions = [
        {
            'transaction_id': f'TXN_BATCH_{i}',
            'source_account': f'ACC_NORMAL_{i}',
            'target_account': f'ACC_NORMAL_{(i+1) % 10}',
            'amount': 30000 + np.random.exponential(10000),
            'timestamp': datetime.utcnow() - timedelta(hours=np.random.randint(0, 24)),
            'mode': np.random.choice(['UPI', 'NEFT']),
        }
        for i in range(5)
    ]
    
    batch_scores = risk_scorer.score_batch(batch_transactions)
    
    logger.info(f"✓ Batch scoring complete ({len(batch_scores)} transactions):")
    for score in batch_scores:
        logger.info(
            f"  - {score.transaction_id}: "
            f"score={score.risk_score:.4f}, "
            f"decision={score.decision}, "
            f"time={score.inference_time_ms:.2f}ms"
        )
    
    # Step 9: Performance Summary
    logger.info("\n[STEP 9] Performance Summary")
    logger.info("-" * 80)
    
    avg_latency = np.mean([s.inference_time_ms for s in batch_scores])
    avg_graph_size = np.mean([s.graph_size for s in batch_scores])
    
    logger.info(f"✓ Performance metrics:")
    logger.info(f"  - Average latency:     {avg_latency:.2f}ms")
    logger.info(f"  - Average graph size:  {avg_graph_size:.0f} nodes")
    logger.info(f"  - Model version:       {fraud_score.model_version}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY: PRODUCTION HTGNN PIPELINE")
    logger.info("=" * 80)
    logger.info("""
Real HTGNN Components:
  ✓ Graph construction from transaction streams
  ✓ Heterogeneous node types (account, device, IP, merchant, phone)
  ✓ Temporal encoding of edges
  ✓ Multi-layer graph attention
  ✓ End-to-end learnable model
  ✓ Comprehensive metrics (ROC-AUC, F1, PR-AUC)
  ✓ Real-time inference
  ✓ Fraud pattern detection

Hybrid Scoring (Real + Heuristics):
  - Graph risk (HTGNN): 60% weight
  - Velocity risk (rules): 20% weight
  - Temporal risk (rules): 15% weight
  - Device risk (rules): 5% weight

Next Steps:
  1. Train on real transaction data
  2. Integrate into FastAPI (see IMPLEMENTATION_GUIDE.md)
  3. Deploy to production
  4. Monitor accuracy and latency
  5. Retrain weekly with new fraud labels
    """)
    
    logger.info("=" * 80)
    logger.info("Example complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()

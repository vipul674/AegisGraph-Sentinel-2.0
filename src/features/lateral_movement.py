import numpy as np
import networkx as nx
from collections import deque, defaultdict


class LateralMovementDetector:
    """
    Detects MITRE ATT&CK TA0008 (Lateral Movement) by tracking
    spikes in Betweenness Centrality across a temporal graph.
    """

    def __init__(
        self,
        history_size=10,
        std_multiplier=2.0,
        spike_multiplier=3.0,
        risk_penalty=0.25
    ):
        self.history_size = history_size
        self.std_multiplier = std_multiplier
        self.spike_multiplier = spike_multiplier
        self.risk_penalty = risk_penalty

        # O(1) append/pop queue to track the last N scores
        self.centrality_history = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        # Active in-memory graph for real-time edge updates
        self.active_graph = nx.DiGraph()

    def update_graph(self, src_account, dst_account):
        """Updates the network topology dynamically."""
        if self.active_graph.has_edge(src_account, dst_account):
            self.active_graph[src_account][dst_account]['weight'] += 1
        else:
            self.active_graph.add_edge(src_account, dst_account, weight=1)

    def _calculate_approx_centrality(self, account_id):
        """Calculates localized betweenness centrality."""
        num_nodes = self.active_graph.number_of_nodes()
        if num_nodes < 3:
            return 0.0

        # Limit the sampling to max 50 nodes for performance
        k_approx = min(50, num_nodes)

        centralities = nx.betweenness_centrality(
            self.active_graph,
            k=k_approx,
            normalized=True,
            weight='weight'
        )
        return centralities.get(account_id, 0.0)

    def analyze_account(self, account_id):
        """Evaluates the account against its historical baseline."""
        current_score = self._calculate_approx_centrality(account_id)
        history = self.centrality_history[account_id]

        if len(history) < 3:
            history.append(current_score)
            return 0.0, False

        baseline_mean = np.mean(history)
        baseline_std = np.std(history)

        # Trigger 1: Current score is above standard deviation threshold
        threshold = baseline_mean + (self.std_multiplier * baseline_std)
        std_trigger = current_score > threshold

        # Trigger 2: Current score is 3x the baseline average
        mult_trigger = current_score > (self.spike_multiplier * baseline_mean)

        is_pivoting = False
        risk_added = 0.0

        # Ignore micro-fluctuations near zero (noise)
        if (std_trigger or mult_trigger) and current_score > 0.05:
            is_pivoting = True
            risk_added = self.risk_penalty

        history.append(current_score)
        return risk_added, is_pivoting


if __name__ == "__main__":
    # Quick Local Test
    detector = LateralMovementDetector()
    print("Initializing Lateral Movement Engine...")

    detector.update_graph("ACC_002", "ACC_001")
    detector.update_graph("ACC_001", "ACC_003")

    detector.centrality_history["ACC_001"].extend(
        [0.01, 0.015, 0.012, 0.01]
    )

    detector.active_graph.add_edges_from(
        [("X", "ACC_001"), ("Y", "ACC_001"), ("ACC_001", "Z")]
    )
    risk, flagged = detector.analyze_account("ACC_001")

    print(f"Risk Penalty Added: {risk} | Flagged as Mule/Pivot: {flagged}")
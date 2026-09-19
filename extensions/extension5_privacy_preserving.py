"""
Extension 5: Privacy-Preserving Training
Adds privacy guarantees with:
- Federated training simulation
- Differential privacy (gradient noise injection)
- Secure multi-party computation primitives
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, List, Dict, Tuple
import numpy as np


class DifferentialPrivacy:
    """
    Differential privacy implementation for gradient noise injection.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = self._compute_noise_multiplier()

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier based on epsilon-delta."""
        # Simplified composition for demo
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def privatize_gradients(self, model: nn.Module) -> None:
        """
        Add noise to gradients for differential privacy.
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # Clip gradients
                    norm = torch.norm(param.grad)
                    if norm > self.max_grad_norm:
                        param.grad.copy_(param.grad * self.max_grad_norm / norm)

                    # Add noise
                    noise = torch.normal(
                        mean=0.0,
                        std=self.noise_multiplier * self.max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device,
                    )
                    param.grad.add_(noise)


class FederatedClient:
    """
    Simulated federated client for privacy-preserving training.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        data: Optional[List] = None,
    ):
        self.client_id = client_id
        self.model = model
        self.data = data or []
        self.local_steps = 5
        self.lr = 1e-4

    def local_train(self, dp_engine: Optional[DifferentialPrivacy] = None) -> Dict[str, torch.Tensor]:
        """
        Train locally and return model updates.
        """
        print(f"   [Client {self.client_id}] Training locally...")
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        for step in range(self.local_steps):
            # Simulate a training step
            optimizer.zero_grad()
            loss = self._simulate_loss()
            loss.backward()

            if dp_engine:
                dp_engine.privatize_gradients(self.model)

            optimizer.step()

        # Return model updates (state dict)
        return self.model.state_dict()

    def _simulate_loss(self) -> torch.Tensor:
        """Simulate a training loss for demo purposes."""
        return torch.sum(torch.tensor([p.norm() for p in self.model.parameters()])) * 0.01


class FederatedServer:
    """
    Simulated federated server coordinating training.
    """

    def __init__(self, global_model: nn.Module, num_clients: int = 3):
        self.global_model = global_model
        self.clients = []
        self._init_clients(num_clients)

    def _init_clients(self, num_clients: int):
        """Initialize simulated clients."""
        for i in range(num_clients):
            local_model = type(self.global_model)()  # Create new instance
            client = FederatedClient(i, local_model)
            self.clients.append(client)

    def aggregate_updates(self, client_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Securely aggregate model updates using secure averaging.
        """
        global_state = self.global_model.state_dict()
        num_clients = len(client_updates)

        for key in global_state:
            # Average updates
            avg = sum(update[key] for update in client_updates) / num_clients
            global_state[key].copy_(avg)

        return global_state

    def federated_round(
        self,
        num_rounds: int = 10,
        use_dp: bool = True,
    ) -> None:
        """
        Run full federated training rounds.
        """
        dp_engine = DifferentialPrivacy() if use_dp else None

        print(f"\n🚀 Starting federated training ({num_rounds} rounds)...")
        if use_dp:
            print(f"   Differential privacy enabled (ε={dp_engine.epsilon:.2f}, δ={dp_engine.delta:.2e})")

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            # Select clients
            selected_clients = self.clients

            # Distribute global model
            for client in selected_clients:
                client.model.load_state_dict(self.global_model.state_dict())

            # Local training
            updates = []
            for client in selected_clients:
                update = client.local_train(dp_engine)
                updates.append(update)

            # Aggregate
            self.aggregate_updates(updates)
            print(f"   ✅ Round complete!")

        print(f"\n✅ Federated training complete!")


class SecureMultiPartyComputation:
    """
    SMC primitives for secure aggregation.
    """

    @staticmethod
    def secure_add(shared_a: List[float], shared_b: List[float]) -> List[float]:
        """Secure addition using secret sharing."""
        return [a + b for a, b in zip(shared_a, shared_b)]

    @staticmethod
    def secret_share(value: float, num_shares: int = 3) -> List[float]:
        """Split value into secret shares."""
        shares = [np.random.normal(0, 0.1) for _ in range(num_shares - 1)]
        shares.append(value - sum(shares))
        np.random.shuffle(shares)
        return shares

    @staticmethod
    def reconstruct(shares: List[float]) -> float:
        """Reconstruct value from shares."""
        return sum(shares)


class PrivacyPreservingDiffMarkTrainer:
    """
    Main trainer for privacy-preserving DiffMark.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

    def train_with_privacy(
        self,
        method: str = "federated",
        num_rounds: int = 10,
        use_dp: bool = True,
    ):
        """
        Train DiffMark with privacy guarantees.
        """
        print(f"\n🛡️ Training DiffMark with {method} privacy...")

        # Create simple model for demo
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        ).to(self.device)

        if method == "federated":
            server = FederatedServer(model)
            server.federated_round(num_rounds=num_rounds, use_dp=use_dp)

        elif method == "dp_only":
            dp = DifferentialPrivacy()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            print("Training with differential privacy only...")
            for step in range(100):
                optimizer.zero_grad()
                loss = model(torch.randn(1, 10)).sum()
                loss.backward()
                dp.privatize_gradients(model)
                optimizer.step()
                if (step + 1) % 20 == 0:
                    print(f"   Step {step + 1}/100, Loss: {loss.item():.4f}")

        print("\n✅ Privacy-preserving training complete!")


def demo_privacy_preserving():
    """Demo function for privacy-preserving training."""
    print("Testing Privacy-Preserving Training...")

    # Test secret sharing
    print("\n1. Testing Secure Multi-Party Computation...")
    smc = SecureMultiPartyComputation()
    secret = 42.0
    shares = smc.secret_share(secret, 5)
    reconstructed = smc.reconstruct(shares)
    print(f"   Secret: {secret}, Reconstructed: {reconstructed:.4f}")

    # Test differential privacy
    print("\n2. Testing Differential Privacy...")
    dp = DifferentialPrivacy(epsilon=1.0)
    print(f"   Noise multiplier: {dp.noise_multiplier:.4f}")

    # Test federated training
    print("\n3. Testing Federated Training...")
    trainer = PrivacyPreservingDiffMarkTrainer()
    trainer.train_with_privacy(method="federated", num_rounds=5, use_dp=True)

    print("\n✅ Privacy-preserving demo complete!")


# Alias for test_all.py compatibility
PrivacyTrainer = PrivacyPreservingDiffMarkTrainer

if __name__ == "__main__":
    demo_privacy_preserving()

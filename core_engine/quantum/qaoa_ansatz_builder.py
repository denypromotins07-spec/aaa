"""
QAOA Ansatz Builder for Portfolio Optimization

Implements the Quantum Approximate Optimization Algorithm (QAOA) with
parameterized quantum circuits for solving QUBO portfolio optimization problems.

This module builds alternating cost and mixer unitaries with support for:
- Custom problem Hamiltonians from QUBO matrices
- Warm-start initialization to avoid barren plateaus
- Multiple mixer types (standard X-mixer, XY-mixer for constraints)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

try:
    import pennylane as qml
    from pennylane import numpy as pnp
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    qml = None
    pnp = np


@dataclass
class QAOAConfig:
    """Configuration for QAOA circuit construction."""
    n_qubits: int
    n_layers: int = 3
    mixer_type: str = "x"  # "x", "xy", or "custom"
    init_strategy: str = "random"  # "random", "warm_start", "uniform"
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if self.mixer_type not in ["x", "xy", "custom"]:
            raise ValueError(f"Unknown mixer_type: {self.mixer_type}")


@dataclass
class QAOAParameters:
    """QAOA variational parameters (gamma, beta) for each layer."""
    gammas: np.ndarray  # Cost Hamiltonian parameters
    betas: np.ndarray   # Mixer Hamiltonian parameters
    
    @classmethod
    def random(cls, n_layers: int, seed: Optional[int] = None) -> 'QAOAParameters':
        """Initialize random parameters in [0, 2π]."""
        rng = np.random.default_rng(seed)
        gammas = rng.uniform(0, 2 * np.pi, n_layers)
        betas = rng.uniform(0, 2 * np.pi, n_layers)
        return cls(gammas=gammas, betas=betas)
    
    @classmethod
    def uniform(cls, n_layers: int, gamma_val: float = 0.0, beta_val: float = 0.0) -> 'QAOAParameters':
        """Initialize all parameters to constant values."""
        return cls(
            gammas=np.full(n_layers, gamma_val),
            betas=np.full(n_layers, beta_val)
        )
    
    def to_array(self) -> np.ndarray:
        """Flatten parameters to single array for optimizers."""
        return np.concatenate([self.gammas, self.betas])
    
    @classmethod
    def from_array(cls, arr: np.ndarray, n_layers: int) -> 'QAOAParameters':
        """Reconstruct parameters from flattened array."""
        if len(arr) != 2 * n_layers:
            raise ValueError(f"Expected {2*n_layers} parameters, got {len(arr)}")
        return cls(
            gammas=arr[:n_layers],
            betas=arr[n_layers:]
        )


class QAOAAnsatzBuilder:
    """
    Builds parameterized QAOA quantum circuits for portfolio optimization.
    
    The QAOA circuit has the form:
    |ψ(γ,β)⟩ = ∏_{l=1}^p exp(-iβ_l H_M) exp(-iγ_l H_C) |+⟩^⊗n
    
    where H_C is the cost Hamiltonian (encoded QUBO) and H_M is the mixer.
    """
    
    def __init__(self, config: QAOAConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.n_layers = config.n_layers
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("Pennylane is required for QAOA. Install with: pip install pennylane")
        
        # Set up device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Store cost Hamiltonian coefficients
        self.cost_coeffs: Dict[Tuple[int, ...], float] = {}
        self.constant_offset: float = 0.0
    
    def set_cost_hamiltonian(self, q_matrix: np.ndarray, linear_terms: Optional[np.ndarray] = None):
        """
        Set the cost Hamiltonian from QUBO matrix.
        
        Args:
            q_matrix: QUBO quadratic coefficient matrix (n x n)
            linear_terms: Linear term vector (n,)
        """
        if q_matrix.shape[0] != self.n_qubits or q_matrix.shape[1] != self.n_qubits:
            raise ValueError(f"Q matrix shape {q_matrix.shape} doesn't match n_qubits={self.n_qubits}")
        
        self.cost_coeffs.clear()
        
        # Add quadratic terms: Q[i,j] * Z_i Z_j
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                coeff = q_matrix[i, j] / 4.0  # Factor from x = (1-Z)/2 transformation
                if abs(coeff) > 1e-10:
                    self.cost_coeffs[(i, j)] = coeff
        
        # Add linear terms from diagonal and explicit linear vector
        for i in range(self.n_qubits):
            coeff = -q_matrix[i, i] / 2.0  # From x = (1-Z)/2
            if linear_terms is not None:
                coeff -= linear_terms[i] / 2.0
            
            if abs(coeff) > 1e-10:
                self.cost_coeffs[(i,)] = coeff
        
        # Calculate constant offset (doesn't affect optimization but useful for energy)
        self.constant_offset = np.sum(q_matrix) / 4.0
        if linear_terms is not None:
            self.constant_offset += np.sum(linear_terms) / 2.0
    
    def build_circuit(self, params: QAOAParameters) -> qml.QNode:
        """
        Build the full QAOA circuit as a PennyLane QNode.
        
        Args:
            params: QAOAParameters with gamma and beta values
            
        Returns:
            PennyLane QNode that implements the QAOA circuit
        """
        if len(params.gammas) != self.n_layers or len(params.betas) != self.n_layers:
            raise ValueError(f"Parameters must have {self.n_layers} layers")
        
        @qml.qnode(self.dev)
        def circuit():
            # Initialize in |+⟩^⊗n state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            # Apply p layers of cost + mixer unitaries
            for layer in range(self.n_layers):
                # Cost unitary: exp(-iγ H_C)
                self._apply_cost_unitary(params.gammas[layer])
                
                # Mixer unitary: exp(-iβ H_M)
                self._apply_mixer_unitary(params.betas[layer])
            
            # Measure expectation value of cost Hamiltonian
            return qml.expval(self._build_cost_hamiltonian())
        
        return circuit
    
    def _apply_cost_unitary(self, gamma: float):
        """Apply the cost unitary exp(-iγ H_C)."""
        for indices, coeff in self.cost_coeffs.items():
            if len(indices) == 1:
                # Single-qubit Z term
                qml.RZ(2 * gamma * coeff, wires=indices[0])
            elif len(indices) == 2:
                # Two-qubit ZZ term
                i, j = indices
                # CNOT-based implementation of exp(-iγ ZZ)
                qml.CNOT(wires=[i, j])
                qml.RZ(2 * gamma * coeff, wires=j)
                qml.CNOT(wires=[i, j])
            else:
                # Higher-order terms (rare in QUBO)
                raise NotImplementedError("Only 1- and 2-body terms supported")
    
    def _apply_mixer_unitary(self, beta: float):
        """Apply the mixer unitary exp(-iβ H_M)."""
        if self.config.mixer_type == "x":
            # Standard X-mixer: H_M = Σ X_i
            for i in range(self.n_qubits):
                qml.RX(2 * beta, wires=i)
        
        elif self.config.mixer_type == "xy":
            # XY-mixer for constrained optimization
            # Preserves Hamming weight
            for i in range(self.n_qubits - 1):
                self._apply_xy_gate(beta, i, i + 1)
        
        elif self.config.mixer_type == "custom":
            # Custom mixer (can be overridden in subclasses)
            self._apply_custom_mixer(beta)
    
    def _apply_xy_gate(self, beta: float, wire1: int, wire2: int):
        """Apply XY interaction gate exp(-iβ(X1X2 + Y1Y2))."""
        # Decomposition using CNOTs and single-qubit gates
        qml.CNOT(wires=[wire1, wire2])
        qml.Hadamard(wires=wire1)
        qml.CNOT(wires=[wire2, wire1])
        qml.RZ(beta, wires=wire1)
        qml.CNOT(wires=[wire2, wire1])
        qml.Hadamard(wires=wire1)
        qml.CNOT(wires=[wire1, wire2])
    
    def _apply_custom_mixer(self, beta: float):
        """Custom mixer implementation (override in subclasses)."""
        # Default to X-mixer
        for i in range(self.n_qubits):
            qml.RX(2 * beta, wires=i)
    
    def _build_cost_hamiltonian(self) -> qml.Hamiltonian:
        """Build PennyLane Hamiltonian object for the cost function."""
        coeffs = []
        observables = []
        
        for indices, coeff in self.cost_coeffs.items():
            if len(indices) == 1:
                obs = qml.Z(indices[0])
            elif len(indices) == 2:
                obs = qml.Z(indices[0]) @ qml.Z(indices[1])
            else:
                continue
            
            coeffs.append(coeff)
            observables.append(obs)
        
        return qml.Hamiltonian(coeffs, observables)
    
    def get_expectation_value(self, params: QAOAParameters) -> float:
        """
        Calculate ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩.
        
        Args:
            params: QAOAParameters
            
        Returns:
            Expectation value of cost Hamiltonian
        """
        circuit = self.build_circuit(params)
        return circuit() + self.constant_offset
    
    def sample_solutions(self, params: QAOAParameters, n_samples: int = 100) -> List[np.ndarray]:
        """
        Sample binary solutions from the QAOA circuit.
        
        Args:
            params: QAOAParameters
            n_samples: Number of samples to draw
            
        Returns:
            List of binary strings (as numpy arrays)
        """
        @qml.qnode(self.dev)
        def sampling_circuit():
            # Same initialization and evolution as build_circuit
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            
            for layer in range(self.n_layers):
                self._apply_cost_unitary(params.gammas[layer])
                self._apply_mixer_unitary(params.betas[layer])
            
            return qml.sample(wires=range(self.n_qubits))
        
        samples = []
        for _ in range(n_samples):
            sample = sampling_circuit()
            samples.append(np.array(sample))
        
        return samples
    
    def get_probabilities(self, params: QAOAParameters) -> np.ndarray:
        """Get probability distribution over all computational basis states."""
        circuit = self.build_circuit(params)
        
        # Get probabilities by measuring in computational basis
        probs = np.zeros(2 ** self.n_qubits)
        
        @qml.qnode(self.dev)
        def prob_circuit():
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            for layer in range(self.n_layers):
                self._apply_cost_unitary(params.gammas[layer])
                self._apply_mixer_unitary(params.betas[layer])
            return qml.probs(wires=range(self.n_qubits))
        
        return prob_circuit()


def create_portfolio_qaoa(
    n_assets: int,
    qubits_per_asset: int = 2,
    n_layers: int = 3,
    warm_start_params: Optional[QAOAParameters] = None,
) -> Tuple[QAOAAnsatzBuilder, QAOAParameters]:
    """
    Create a QAOA ansatz for portfolio optimization.
    
    Args:
        n_assets: Number of assets in portfolio
        qubits_per_asset: Qubits per asset for weight discretization
        n_layers: Number of QAOA layers (circuit depth)
        warm_start_params: Optional initial parameters from classical solution
        
    Returns:
        Tuple of (QAOAAnsatzBuilder, initial QAOAParameters)
    """
    n_qubits = n_assets * qubits_per_asset
    
    config = QAOAConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        mixer_type="x",
        init_strategy="warm_start" if warm_start_params is not None else "random",
    )
    
    builder = QAOAAnsatzBuilder(config)
    
    if warm_start_params is not None:
        initial_params = warm_start_params
    else:
        initial_params = QAOAParameters.random(n_layers, seed=42)
    
    return builder, initial_params


if __name__ == "__main__":
    # Example usage
    print("QAOA Ansatz Builder for Portfolio Optimization")
    print("=" * 50)
    
    if not PENNYLANE_AVAILABLE:
        print("Pennylane not available. Install with: pip install pennylane")
    else:
        # Create a simple 2-asset, 2-qubit-per-asset example
        builder, params = create_portfolio_qaoa(
            n_assets=2,
            qubits_per_asset=2,
            n_layers=2,
        )
        
        # Set up a simple cost Hamiltonian
        q_matrix = np.array([
            [1.0, -0.5, 0.0, 0.0],
            [-0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -0.3],
            [0.0, 0.0, -0.3, 1.0],
        ])
        
        builder.set_cost_hamiltonian(q_matrix)
        
        # Calculate expectation value
        energy = builder.get_expectation_value(params)
        print(f"Initial energy: {energy:.4f}")
        
        # Sample some solutions
        samples = builder.sample_solutions(params, n_samples=10)
        print(f"\nSampled solutions:")
        for i, s in enumerate(samples[:5]):
            print(f"  {i}: {s}")

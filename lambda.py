from functions import *
from quantum_circuit_classes import *

np.set_printoptions(precision=3, suppress=True, linewidth=100)

print("Number of qubits for the input register: ")
N = int(input())
M = 2 * N

initial_state = np.array([0]*(N+M)) # Initial state all ket 0
initial_state_vector = stateToVector(initial_state)

# Random complex vector v
complex_numbers = np.random.normal(size=2**N) + 1j *  np.random.normal(size=2**N)
complex_numbers /= np.linalg.norm(complex_numbers, ord=2)
coefficients = complex_numbers.tolist()

print(f"Coefficients: {coefficients}")

alphas, thetas = qspParameters(coefficients, N)

# Association of alphas and binary strings
basis_vectors = compBasis(N)[1:]
alpha_vector = {basis_vectors[idx]: coeff for idx, coeff in enumerate(alphas[N-2])}

circuit = QuantumCircuit(N + M, initial_state_vector)

pcs = PrefixCopyStage(N, M)
circuit.addStage(pcs)

gis = GrayInitialStage(N, M, alpha_vector)
circuit.addStage(gis)

scs = SuffixCopyStage(N, M)
circuit.addStage(scs)

gps = GrayPathStage(N, M, alpha_vector)
circuit.addStage(gps)

invs = InverseStage(N, M)
circuit.addStage(invs)

circuit.printCircuit(mode="figure", modulo=False)
#circuit.printCircuit(mode="console", modulo=False)
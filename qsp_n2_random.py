from functions import *
from quantum_circuit_classes import *

np.set_printoptions(precision=3, suppress=True, linewidth=100)

N = 2
M = 2 * N

initial_state = np.array([0]*(N+M)) # Initial state all ket 0
initial_state_vector = stateToVector(initial_state)

complex_numbers = np.random.normal(size=2**N) + 1j *  np.random.normal(size=2**N)
complex_numbers /= np.linalg.norm(complex_numbers, ord=2)
coefficients = complex_numbers.tolist()

new_qubits = [0] * M
state_vector = stateToVector(new_qubits)
coefficients = np.kron(coefficients, state_vector)

print(f"Coefficients: {coefficients}")
print(f"Modulo: {[np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in coefficients]}")

alphas, thetas = qspParameters(coefficients, N)

circuit = QuantumCircuit(N + M, initial_state_vector)

yr = RotY(1, thetas[0], 0)
circuit.addStage(yr)

basis_vectors = compBasis(N)[1:]
alpha_vector = {basis_vectors[idx]: coeff for idx, coeff in enumerate(alphas[N-2])}

pre = UnitPre(N, M, N-1)
circuit.addStage(pre)

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

post = UnitPost(N, M, N-1)
circuit.addStage(post)

#circuit.printCircuit(mode="state", modulo=True)
#circuit.printCircuit(mode="matrix", modulo=True)
circuit.printCircuit(mode="ket", modulo=True)
circuit.printCircuit(mode="figure", modulo=False)
#circuit.printCircuit(mode="console", modulo=False)

print(f"Fidelity of the states: {qml.math.fidelity_statevector([np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in coefficients], circuit.getState(modulo=True))}")

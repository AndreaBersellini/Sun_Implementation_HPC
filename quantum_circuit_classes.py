from functions import *
from operator import add
from math import floor, log2
import matplotlib.pyplot as plt

class Stage:
    def __init__(self, qubits : int, ancillaries : int, offset : int):
        self._qubits = qubits
        self._ancillaries = ancillaries
        self._offset = offset
        if ancillaries != 0 : self._t = int(floor(log2(self._ancillaries / 2)))

    def circuit(self) -> list:
        raise "Abstract method!"
    
class PrefixCopyStage(Stage):
    def __init__(self, qubits : int, ancillaries : int, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)

    def circuit(self) -> list:
        # COPY REGISTER INITIALIZATION ("k" COPIES OF THE FIRST "t" QUBITS)
        k = int(floor(self._ancillaries / (2 * self._t)))
        
        for x in range(self._t):
            control, target = x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        for i in range(k - 1):
            for x in range(self._t):
                control, target = (self._t * i) + self._qubits + x, self._t + (self._t * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])

class GrayInitialStage(Stage):
    def __init__(self, qubits : int, ancillaries : int, alphas : list, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)
        self._alphas = alphas

    def circuit(self) -> list:
        binary = ['0' * (self._qubits - self._t)] * pow(2, self._t) # Is the same of calculating the first element of Gray code 1 and 2 (00...00)
        basis = [s[::-1] for s in compBasis(self._t)]
        strings = list(map(add, basis, binary)) # Strings in the first column have the last "(n − t)" bits at 0, and strings in each row share the same first "t" bits

        # PHASE REGISTER INITIALIZATION (NON PARALLELIZZABILE!)
        for i, string in enumerate(strings):
            for j, bit in enumerate(string):
                if bit == '1':
                    control, target = self._qubits + j, self._qubits * 2 + i
                    qml.CNOT(wires = [control + self._offset, target + self._offset])

        # ALPHA ROTATIONS
        for i, id in enumerate(strings):
            if '1' in id:
                wire = self._qubits * 2 + i
                qml.PhaseShift(self._alphas[id], wires = [wire + self._offset])

class SuffixCopyStage(Stage):
    def __init__(self, qubits : int, ancillaries : int, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)

    def circuit(self) -> list:
        # COPY REGISTER RESET (INVERSE OF PREFIX COPY STAGE)
        k = int(floor(self._ancillaries / (2 * self._t)))

        for i in range(k-1):
            for x in range(self._t)[::-1]:
                control, target = (self._t * i) + self._qubits + x, self._t + (self._t * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])

        for x in range(self._t)[::-1]:
            control, target = x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        # COPY REGISTER INITIALIZATION ("k" COPIES OF THE LAST "n - t" QUBITS)
        k = int(floor(self._ancillaries / (2 * (self._qubits - self._t))))

        for x in range(self._qubits - self._t):
            control, target = self._t + x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        for i in range(k - 1):
            for x in range(self._qubits - self._t):
                control, target = ((self._qubits - self._t) * i) + self._qubits + x, (self._qubits - self._t) + ((self._qubits - self._t) * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])

class GrayPathStage(Stage):
    def __init__(self, qubits : int, ancillaries : int, alphas : list, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)
        self._alphas = alphas

    def circuit(self) -> list:
        gray_1 = grayCode(1, self._qubits - self._t)
        gray_2 = grayCode(2, self._qubits - self._t)

        basis = [s[::-1] for s in compBasis(self._t)]
        binary = ['0' * (self._qubits - self._t)] * pow(2, self._t)

        prev = list(map(add, basis, binary)) # Vector of Gray Initial Stage binary strings

        for k in range(int(pow(2, self._qubits) / pow(2, self._t) - 1)):

            curr = [] # Current phase string vector

            for i, id in enumerate(basis):
                g1 = str("".join(map(str, gray_1[k + 1])))
                g2 = str("".join(map(str, gray_2[k + 1])))
                curr.append(id + g1 if i < (len(basis) / 2) else id + g2)

            # PHASE REGISTER INITIALIZATION (NON PARALLELIZZABILE!)
            for i, id in enumerate(basis):
                curr_str = curr[i]
                prev_str = prev[i]
                change_vector = [True if b1 != b2 else False for b1, b2 in zip(prev_str, curr_str)]

                for bit, change in enumerate(change_vector[self._t:]):
                    if change:
                        control, target = self._qubits + bit, self._qubits * 2 + i
                        qml.CNOT(wires = [control + self._offset, target + self._offset])

            # ALPHA ROTATIONS
            for i, id in enumerate(basis):
                string = curr[i]
                wire = self._qubits * 2 + i
                qml.PhaseShift(self._alphas[string], wires = [wire + self._offset])

            prev = curr # Replace previous phase with current phase
        
class InverseStage(Stage):
    def __init__(self, qubits : int, ancillaries : int, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)

    def circuit(self) -> list:
        # INVERSE GRAY PATH STAGE
        gray_1 = grayCode(1, self._qubits - self._t)
        gray_2 = grayCode(2, self._qubits - self._t)

        basis = [s[::-1] for s in compBasis(self._t)]
        binary = ['0' * (self._qubits - self._t)] * pow(2, self._t)
        strings = [list(map(add, basis, binary))]

        for k in range(int(pow(2, self._qubits) / pow(2, self._t) - 1)):

            s = []

            for i, id in enumerate(basis):
                g1 = str("".join(map(str, gray_1[k + 1])))
                g2 = str("".join(map(str, gray_2[k + 1])))
                s.append(id + g1 if i < (len(basis) / 2) else id + g2)

            strings.append(s)
        
        strings = strings[::-1]

        for k in range(int(pow(2, self._qubits) / pow(2, self._t) - 1))[::-1]:

            curr = strings[k]
            post = strings[k + 1]

            for i in range(len(basis))[::-1]:
                curr_str = curr[i]
                post_str = post[i]
                change_vector = [True if b1 != b2 else False for b1, b2 in zip(post_str, curr_str)]

                for bit, change in enumerate(change_vector[self._t:]):
                    if change:
                        control, target = self._qubits + bit, self._qubits * 2 + i
                        qml.CNOT(wires = [control + self._offset, target + self._offset])

        # INVERSE SUFFIX COPY STAGE
        k = int(floor(self._ancillaries / (2 * (self._qubits - self._t))))

        for i in range(k - 1):
            for x in range(self._qubits - self._t)[::-1]:
                control, target = ((self._qubits - self._t) * i) + self._qubits + x, (self._qubits - self._t) + ((self._qubits - self._t) * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])
                
        for x in range(self._qubits - self._t)[::-1]:
            control, target = self._t + x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        k = int(floor(self._ancillaries / (2 * self._t)))
        
        for x in range(self._t):
            control, target = x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        for i in range(k - 1):
            for x in range(self._t):
                control, target = (self._t * i) + self._qubits + x, self._t + (self._t * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])

        # INVERSE GRAY INITIAL STAGE
        strings = list(map(add, basis, binary))

        for i, string in enumerate(strings[::-1]):
            for j, bit in enumerate(string[::-1]):
                if bit == '1':
                    control, target = self._qubits + len(string) - 1 - j, self._qubits * 2 + len(strings) - 1 - i
                    qml.CNOT(wires = [control + self._offset, target + self._offset])

        # INVERSE PREFIX COPY STAGE
        for i in range(k-1):
            for x in range(self._t)[::-1]:
                control, target = (self._t * i) + self._qubits + x, self._t + (self._t * i) + self._qubits + x
                qml.CNOT(wires = [control + self._offset, target + self._offset])

        for x in range(self._t)[::-1]:
            control, target = x, self._qubits + x
            qml.CNOT(wires = [control + self._offset, target + self._offset])

        #for i in range(self._ancillaries):
        #    qml.WireCut(wires=self._qubits + i)

class RotY(Stage):
    def __init__(self, qubits : int, angle : float, wire : int, offset : int = 0):
        Stage.__init__(self, qubits, 0, offset)
        self._angle = angle
        self._wire = wire

    def circuit(self) -> list:
        qml.RY(2*self._angle, self._wire + self._offset)

class UnitPre(Stage):
    def __init__(self, qubits : int, ancillaries : int, wire : int, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)
        self._wire = wire

    def circuit(self) -> list:
        qml.adjoint(qml.S)(self._wire)
        qml.Hadamard(self._wire + self._offset)

class UnitPost(Stage):
    def __init__(self, qubits : int, ancillaries : int, wire : int, offset : int = 0):
        Stage.__init__(self, qubits, ancillaries, offset)
        self._wire = wire

    def circuit(self) -> list:
        qml.Hadamard(self._wire)
        qml.S(self._wire + self._offset)

class ControlGate(Stage):
    def __init__(self, qubits : int, angle : float, wire : int, cw, offset : int = 0):
        Stage.__init__(self, qubits, 0, offset)
        self._angle = angle
        self._wire = wire
        self._cw = cw

    def circuit(self) -> list:
        U = np.matrix([[np.cos(self._angle),-np.sin(self._angle)],[np.sin(self._angle),np.cos(self._angle)]])
        qml.ControlledQubitUnitary(U, self._wire-1, self._wire, self._cw)
        #qml.RY(2*self._angle, self._wire + self._offset)

class QuantumCircuit():
    def __init__(self, wires : int, state_vector : np.ndarray):
        self._wires = wires
        self._input_register = state_vector

        self._stages = []

        self._device = qml.device('default.qubit', wires)
    
    def addStage(self, stage):
        self._stages.append(stage)

    def circuit(self):
        for s in self._stages:
            s.circuit()
        return qml.expval(qml.PauliZ(0))
    
    def circuitDisplay(self, state_vector : list, n : int):
        #qml.BasisEmbedding(state_vector, range(n))
        qml.QubitStateVector(state_vector, range(n))
        for s in self._stages:
            s.circuit()
        return qml.state()
    
    def circuitState(self, state_vector: list, n: int):
        @qml.qnode(self._device)
        def _circuit():
            #qml.BasisEmbedding(state_vector, wires=range(n))
            qml.QubitStateVector(state_vector, range(n))
            for s in self._stages:
                s.circuit()
            return qml.state()
        return _circuit()

    def printCircuit(self, mode : str, modulo : bool = False) -> None:
        qml.QNode(self.circuit, self._device)
        match mode:
            case "console":
                print(qml.draw(self.circuitDisplay, decimals = 4)(self._input_register, self._wires))

            case "figure":
                qml.draw_mpl(self.circuitDisplay, decimals = 4)(self._input_register, self._wires)
                plt.savefig(f"lambda_{self._wires//3}.png", dpi=300, bbox_inches="tight")
                plt.show()

            case "ket":
                matrix = qml.matrix(self.circuit, wire_order = list(range(self._wires)))()
                state = matrix @ self._input_register
                if modulo == True:
                    state = [np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in state]
                print(toKet(state, self._wires))

            case "matrix":
                matrix = qml.matrix(self.circuit, wire_order = list(range(self._wires)))()
                if modulo == True:
                    matrix = [np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in matrix]

                with open("circuit_matrix.txt", "w") as f:
                    for row in matrix:
                        f.write(", ".join(map(str, row)) + "\n")
                #print(matrix)

            case "state":
                matrix = qml.matrix(self.circuit, wire_order = list(range(self._wires)))()
                sv = matrix @ self._input_register
                if modulo == True:
                    sv = [np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in sv]
                print(sv)
    
    def getState(self, modulo : bool = False) -> np.ndarray:
        matrix = qml.matrix(self.circuit, wire_order = list(range(self._wires)))()
        sv = matrix @ self._input_register
        if modulo == True:
            sv = [np.sqrt(np.real(x)**2 + np.imag(x)**2) for x in sv]
        return sv
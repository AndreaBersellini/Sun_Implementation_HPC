from math import sqrt, acos
import pennylane as qml
import numpy as np

def binaryInnProd(x : int, y : int, n : int) -> str: # Compute the inner product in F^2 between two states of 'n' qubits
    v = []
    for i in range(n):
        v.append(int(x[i]) * int(y[i]))

    return bin(sum(v))[-1]

def compBasis(n: int) -> str: # Generate the computational basis vectors for 'n' qubits
    strings = []

    for k in range(pow(2, n)):
        strings.append(bin(k)[2:].zfill(n))

    return strings

def toKet(vector : list, n : int) -> str: # Show the KET notation of an input vector of 'n' qubits
    basis = compBasis(n)
    state = ''

    for i, q in enumerate(vector):
        if q != 0:
            state += (str(q) + '|' + str(basis[i]) + '>  ')

    return state

def grayCode(n_gray : int, n : int):
    up_prefix = np.array([[0]])
    down_prefix = np.array([[1]])
    matrix = np.concatenate((up_prefix, down_prefix))

    match n_gray:
        case 1:
            for _ in range(n - 1):
                up_prefix = np.concatenate((up_prefix, up_prefix))
                down_prefix = np.concatenate((down_prefix, down_prefix))
                matrix = np.concatenate((np.concatenate((matrix, np.flip(matrix, 0))), np.concatenate((up_prefix, down_prefix))), axis = 1)
        case 2:
            for _ in range(n - 1):
                up_prefix = np.concatenate((up_prefix, up_prefix))
                down_prefix = np.concatenate((down_prefix, down_prefix))
                matrix = np.concatenate((np.concatenate((up_prefix, down_prefix)), np.concatenate((matrix, np.flip(matrix, 0)))), axis = 1)
            
    return matrix

class Node:
    def __init__(self, value, arc):
        self._value = value # Value of the node
        self._arc_val = arc # Value of the arc from node to parent
        self._children = [] # Children nodes

    def addChild(self, node: 'Node'): # Add a brach to the tree
        self._children.append(node)
    
    def nodeVal(self) -> float: # Value of the node
        return self._value
    
    def arcVal(self) -> str: # Value of the arc
        return self._arc_val
    
    def explore(self, string : str) -> float: # Explore recursively the tree until the desire state
        if not string:
            return self.nodeVal()
        
        token = string[0]
        for childe in self._children:
            if childe.arcVal() == token:
                return childe.explore(string[1:])
    
    def printTree(self, depth : int = 0):
        for childe in self._children:
            childe.printTree(depth + 1)
        ind = '-' * depth
        print(ind + str(self.nodeVal()))

def generateBinTree(vector_v : list, bit : str, k : int): # Generate a binary search tree of depth 'k' from a list of coeffiecients
    value = np.linalg.norm(vector_v, ord = 2) # Norm-2
    node = Node(value, bit)

    if k != 0:
        vector0 = vector_v[:len(vector_v)//2]
        vector1 = vector_v[len(vector_v)//2:]

        node.addChild(generateBinTree(vector0, '0', k - 1))
        node.addChild(generateBinTree(vector1, '1', k - 1))

    return node

def generateThetaVect(unit_vector : np.ndarray, n : int): # Generate the vector of theta from a multiplexor rappresentation in binary strings
    theta = np.array([])

    binary_strings = encodeMultiplexor(np.array(['0']), n)
    bst = generateBinTree(unit_vector, None, n)
    #bst.printTree()

    for child_string in binary_strings:
        
        parent_string = child_string[:-1]
        child = bst.explore(child_string)
        parent = bst.explore(parent_string)

        theta = np.append(theta, acos(child / parent))

    return theta

def generateInnProdMatrix(n: int) -> np.ndarray: # Generate the matrix of inner products between every computational basis vectors except |00...00>
    basis_vectors = compBasis(n)[1:]
    coefficients = np.array([])

    for i in basis_vectors:
        row = np.array([])
        for j in basis_vectors:
            row = np.append(row, float(binaryInnProd(i, j, n)))
        coefficients = np.append(coefficients, row)
    coefficients = np.reshape(coefficients, (len(basis_vectors), len(basis_vectors)))

    return coefficients

def generateAlphaVect(coeffiecients : np.ndarray, theta_vector : np.ndarray) -> np.ndarray: # Generate the vector of alpha by solving the linear system
    alpha_vector = np.linalg.solve(coeffiecients, theta_vector)
    return alpha_vector

def encodeMultiplexor(binary_vector : np.ndarray, n : int) -> np.ndarray: # Generate all the binary strings to use for the exploration of the binary tree
    n = n - 1

    if n == 0:
        return binary_vector
    
    elem_0 = '0' + binary_vector[-1]
    branch_0 = np.append(binary_vector, elem_0)
    branch_0 = encodeMultiplexor(branch_0, n)

    elem_1 = '1' + binary_vector[-1]
    branch_1 = np.append(binary_vector, elem_1)
    branch_1 = encodeMultiplexor(branch_1, n)

    binary_vector = np.union1d(branch_0, branch_1)

    return binary_vector

def classicalAlgorithm(unit_vector : np.ndarray, n : int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    assert len(unit_vector) != (pow(2, n) - 1), "Wrong number of coefficients 'V' or wrong value of 'n'."

    theta_vector = generateThetaVect(unit_vector, n)

    binary_matrix = generateInnProdMatrix(n)

    alpha_vector = generateAlphaVect(binary_matrix, theta_vector)

    alpha_check = (pow(2, 1 - n) * (2 * binary_matrix - np.ones((pow(2, n) - 1,pow(2, n) - 1))) @ theta_vector) # (2^(1-n))*(2*A - J)*thetas
    assert np.allclose(alpha_check, alpha_vector, atol = 1e-8), "Alpha values don't match the check: \n" + str(alpha_vector) + " -> " + str(alpha_check)

    return (alpha_vector, theta_vector, binary_matrix)

def qspParameters(unit_vector : np.ndarray, n : int) -> tuple[list, np.ndarray]:

    assert len(unit_vector) != (pow(2, n) - 1), "Wrong number of coefficients 'V' or wrong value of 'n'."

    theta_vector = generateThetaVect(unit_vector, n) # [(2^n)-1] theta angles
    
    alpha_vector = [] # Cannot use numpy because the resulting array has different shapes

    print(f"Thetas generated from BST: {theta_vector}")
    print("_"*50)
    #theta_vector = [th/2 for th in theta_vector]

    for k in range(n): # Multiplexor decomposition
        diag_rz = np.array([])
        for p in range(pow(2, k)):
            th = theta_vector[pow(2, k) + p - 1]
            diag = np.array([np.exp(-1j*(th)), np.exp(1j*(th))])
            diag_rz = np.append(diag_rz, diag)
        
        print(f"Diagonal Rz(th) of UCG {k+1}: {diag_rz}")
        #print(f"Thetas for diag(Rz(th_i)) of UCG {k+1}: {[np.angle(x) for x in diag_rz]}")

        if k != 0:
            lamb = np.array([coeff / diag_rz[0] for coeff in diag_rz])
            print(f"Exponential diagonal of Lambda {k+1}: {lamb}")

            comb = [np.angle(x) for x in lamb]
            #print(f"Angle combination for UCG {k+1}: {comb}")

            binary_matrix = generateInnProdMatrix(k+1)
            alphas = generateAlphaVect(binary_matrix, comb[1:])
            #print(f"mat {k}: {binary_matrix}")
            print(f"Aphas for Lambda {k+1}: {alphas}")

            alpha_check = (pow(2, 1 - (k + 1)) * (2 * binary_matrix - np.ones((pow(2, (k + 1)) - 1,pow(2, (k + 1)) - 1))) @ comb[1:]) # (2^(1-n))*(2*A - J)*thetas

            assert np.allclose(alpha_check, alphas, atol = 1e-8), "Alpha values don't match the check: \n" + str(alphas) + " -> " + str(alpha_check)

            alpha_vector.append(alphas)

        print("_"*50)

    return (alpha_vector, theta_vector)

def stateToVector(state : np.ndarray):
    state_vector = 1
    for ks in state:
        vector = np.array([1 - ks, ks]) # Vector notation of ket state
        state_vector = np.kron(state_vector, vector)
    return state_vector
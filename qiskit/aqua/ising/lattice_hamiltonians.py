from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import copy
import random
import itertools
import numpy as np
from qiskit import Aer


from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator, QuantumInstance

from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.qiskiterror import QiskitError






def generate_lattice_hamiltonian(problem_type,length,t,J):

    if problem_type == 'isi':



        pauli_list = []
        # ZZ interactions
        for site in range(length - 1):
            pauli_v = np.zeros(length)
            pauli_w = np.zeros(length)
            pauli_v[site] = 1
            pauli_v[site + 1] = 1
            pauli_list.append([J, Pauli(pauli_v, pauli_w)])
        # Transverse field
        for site in range(length):
            pauli_v = np.zeros(length)
            pauli_w = np.zeros(length)
            pauli_w[site] = 1
            pauli_list.append([t, Pauli(pauli_v, pauli_w)])

        hamiltonian = Operator(pauli_list)


    elif problem_type == 'hei':


        pauli_list = []
        # ZZ interactions
        for site in range(length - 1):
            pauli_v = np.zeros(length)
            pauli_w = np.zeros(length)
            pauli_v[site] = 1
            pauli_v[site + 1] = 1
            pauli_list.append([J, Pauli(pauli_v, pauli_w)])
        # XX interaction
        for site in range(length - 1):
            pauli_v = np.zeros(length)
            pauli_w = np.zeros(length)
            pauli_w[site] = 1
            pauli_w[site + 1] = 1
            pauli_list.append([t, Pauli(pauli_v, pauli_w)])

        # YY interaction
        for site in range(length - 1):
            pauli_v = np.zeros(length)
            pauli_w = np.zeros(length)
            pauli_w[site] = 1
            pauli_w[site + 1] = 1
            pauli_v[site] = 1
            pauli_v[site + 1] = 1
            pauli_list.append([t, Pauli(pauli_v, pauli_w)])

        hamiltonian = Operator(pauli_list)

    else:
        raise QiskitError('Lattice model not supported.')

    return hamiltonian


def print_grouped_hamiltonian(hamiltonian):
    threshold=10**-13
    if hamiltonian._grouped_paulis is not None:
        for idx, tpb_set in enumerate(hamiltonian.grouped_paulis):
            for elem in tpb_set:
                if (abs(elem[0])>threshold): print(elem[0],str(elem[1]), idx, '')
    elif hamiltonian._paulis is not None:
        for idx, pauli in enumerate(hamiltonian.paulis):
                if (abs(pauli[0])>threshold): print(pauli[0],str(pauli[1]), idx, '')
    else:
        print("Not Using Paulis / Grouped Paulis")

def print_basis(hamiltonian):

    if hamiltonian._grouped_paulis is not None:
        n_qubits=hamiltonian.num_qubits
        print('\nbasis of grouped paulis')
        baseslist=[]
        for idx, tpb_set in enumerate(hamiltonian.grouped_paulis):
            print('group:',idx)
            list=[]
            for qubit_idx in range(n_qubits):
                if tpb_set[0][1].x[qubit_idx]:
                    if tpb_set[0][1].z[qubit_idx]:
                        list.append('Y')
                    else:
                        list.append('X')
                else:
                    if tpb_set[0][1].z[qubit_idx]:
                        list.append('Z')
                    else:
                        list.append('Z') #considered as basis I ->Z
            print(list)
            baseslist.append(list)
    elif hamiltonian._paulis is not None:
        n_qubits=hamiltonian.num_qubits
        print('\nbasis of paulis')
        baseslist=[]
        for idx, pauli in enumerate(hamiltonian.paulis):
            print('group:',idx)
            list=[]
            for qubit_idx in range(n_qubits):
                if pauli[1].x[qubit_idx]:
                    if pauli[1].z[qubit_idx]:
                        list.append('Y')
                    else:
                        list.append('X')
                else:
                    if pauli[1].z[qubit_idx]:
                        list.append('Z')
                    else:
                        list.append('Z')  #considered as basis I ->Z
            print(list)
            baseslist.append(list)
    else:
        print("Using Paulis")
        baseslist=[]

    return baseslist

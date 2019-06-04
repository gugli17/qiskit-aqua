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

from qiskit.aqua.components.variational_forms.ry import RY
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.utils.backend_utils import is_statevector_backend

from qiskit.visualization import *


#######GEN HAMILTONIAN ##############

problem_type = 'isi'
length = 2
n_qubits=length

print('\nGenerate hamiltonian in Pauli format...')

if problem_type == 'isi':


    t = -1.0
    J = 1.0
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

    Jz = 1.0
    Jxy = 1.0
    pauli_list = []
    # ZZ interactions
    for site in range(length - 1):
        pauli_v = np.zeros(length)
        pauli_w = np.zeros(length)
        pauli_v[site] = 1
        pauli_v[site + 1] = 1
        pauli_list.append([Jz, Pauli(pauli_v, pauli_w)])
    # XX interaction
    for site in range(length - 1):
        pauli_v = np.zeros(length)
        pauli_w = np.zeros(length)
        pauli_w[site] = 1
        pauli_w[site + 1] = 1
        pauli_list.append([Jxy, Pauli(pauli_v, pauli_w)])

    # YY interaction
    for site in range(length - 1):
        pauli_v = np.zeros(length)
        pauli_w = np.zeros(length)
        pauli_w[site] = 1
        pauli_w[site + 1] = 1
        pauli_v[site] = 1
        pauli_v[site + 1] = 1
        pauli_list.append([Jxy, Pauli(pauli_v, pauli_w)])

    hamiltonian = Operator(pauli_list)

else:
    raise QiskitError('Lattice model not supported.')


'Hamiltonian is written using Paulis, so the current repr. is Paulis'
print('print what i have generated')
print(hamiltonian.print_operators())
print('\nCurrent repr:',hamiltonian.representations)




print('\nDiagonalize it..')
exact_eigensolver = ExactEigensolver(hamiltonian, k=1)
ret = exact_eigensolver.run()
print('The computed energy is: {:.12f}'.format(ret['eigvals'][0].real))
'Hamitlonian is implicitly converted into matrix in the ED routine, so this is the current stored repr.'
print('\nCurrent repr:',hamiltonian.representations)



'Hamitlonian to have it in groped paulis, call the _check_representation function'
hamiltonian._check_representation('grouped_paulis')
print('\nCurrent repr:',hamiltonian.representations)
print('grouped:\n', hamiltonian.grouped_paulis)


'Visualize the groups'

for group in range(len(hamiltonian.grouped_paulis)):
    print('group:',group)
    list=hamiltonian.grouped_paulis[group]
    for elem in list:
        print(elem[0],str(elem[1]))


'visualize the basis'
print('\nbasis of grouped paulis')
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
                list.append('I')
    print(list)
print(hamiltonian)

#######GEN CIRCUIT ##############

initialstate = Zero(n_qubits)
depth=1
var_form = RY(n_qubits,depth,initial_state=initialstate)
param = np.zeros(n_qubits*(depth+1))
for i in range(len(param)):
    param[i]=i
#print(varform.construct_circuit(param))
#backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')
circuit = var_form.construct_circuit(param)

########GEN CIRCUIT + Measure?

operator_mode='grouped_paulis'
operator_mode='paulis'
#operator_mode='matrix'
input_circuit = circuit

max_eval = 100
number_of_shots =2048
cobyla = COBYLA(maxiter=max_eval)

print(circuit)



vqe = VQE(hamiltonian, var_form, optimizer=cobyla, operator_mode=operator_mode,initial_point=None, aux_operators=None, callback=None)
quantum_instance=QuantumInstance(backend=backend, shots = number_of_shots)
results = vqe.run(quantum_instance)
print('The computed ground state energy is: {:.12f}'.format(results['energy']))
print("Parameters: {}".format(results['opt_params']))
print(results)

print(vqe.get_optimal_circuit())

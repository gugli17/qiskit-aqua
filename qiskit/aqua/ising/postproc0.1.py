from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from copy import deepcopy
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
from qiskit.aqua.utils import compile_and_run_circuits
from qiskit.aqua.utils import find_regs_by_name
from qiskit.aqua.utils.backend_utils import is_statevector_backend
from qiskit.visualization import *


from lattice_hamiltonians import *




t=1.1
J=1.0
problem_type='hei'
L=2
#backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')

hamiltonian = generate_lattice_hamiltonian(problem_type,L,t,J)

hamiltonian._check_representation('grouped_paulis')

'Visualize the groups'
print_grouped_hamiltonian(hamiltonian)
bases=print_basis(hamiltonian)


'Initialize ancilla register + classical register'
q = QuantumRegister(L, name='q')
qa = QuantumRegister(L, name='qa')
cr = ClassicalRegister(2*L, name='cr')
cv = ClassicalRegister(L, name='cv')

system_circuit = QuantumCircuit(qa,cr)


'Create VQE circuit and execute'
print('WF circuit alone...')


initialstate = Zero(L)
depth=1
var_form = RY(L,depth,initial_state=initialstate)
param = np.zeros(L*(depth+1))
for i in range(len(param)):
    param[i]=i

vqe_circuit = var_form.construct_circuit(param)

'''
dummy_circ = QuantumCircuit(q)
for i in range(L):
    dummy_circ.h(q[i])
dummy_circ.barrier()
vqe_circuit = dummy_circ
'''

vqe_alone = QuantumCircuit()
vqe_alone.add_register(cv)

vqe_alone += vqe_circuit
vqe_alone.measure(q,cv)
print(vqe_alone)
result = execute(vqe_alone,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())






'Merge VQE circuit with ancilla and classical register total one'
q_vqe = find_regs_by_name(vqe_circuit, 'q')
system_circuit += vqe_circuit




'Create entangled copy'
for i in range(L):
    system_circuit.cx(q_vqe[i],qa[i])

system_circuit.barrier()

#print(system_circuit)




'Generate measurement circuits'
print('\nGenerate full circuits...\n')

circuits = []





for i in range(len(bases)):
    circ = QuantumCircuit() + system_circuit
    q = find_regs_by_name(circ, 'q')
    #c = find_regs_by_name(circ, 'cr', qreg=False)
    print(bases[i])
    for j in range(L):
        rot = bases[i][j]
        if (rot == 'X'): circ.h(q[j])
        #elif (rot == 'Y'): circ.u3(np.pi, np.pi/2, np.pi/2, q[j])
        elif (rot == 'Y'):
            circ.u1(np.pi/2, q[j]).inverse()  # s
            circ.u2(0.0, np.pi, q[j])  # h

    circ.barrier()
    for i in range(2*L):
        circ.measure(i,i)

    print(circ)
    circuits.append(circ)



for i in range(len(circuits)):

    circ = circuits[i]
    print('#', bases[i])
    result = execute(circ,backend).result()
    if is_statevector_backend(backend): print(result.get_statevector())
    else: print(result.get_counts())

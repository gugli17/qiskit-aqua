from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from copy import deepcopy
import random
import itertools
import numpy as np
from qiskit import Aer

from qiskit import BasicAer

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
from qmc03 import *



t=-1.0
J=-1.0
problem_type='isi'
L=3
#backend = Aer.get_backend('statevector_simulator')
#representation='matrix'
backend = Aer.get_backend('qasm_simulator')
representation='grouped_paulis'
#representation='paulis'

hamiltonian = generate_lattice_hamiltonian(problem_type,L,t,J)


print('\nDiagonalize it..')
exact_eigensolver = ExactEigensolver(hamiltonian, k=1)
ret = exact_eigensolver.run()
print('The computed energy is: {:.12f}'.format(ret['eigvals'][0].real))
print('Eigenvector is:', ret['wavefunction'])
'Hamitlonian is implicitly converted into matrix in the ED routine, so this is the current stored repr.'
#print(hamiltonian.print_operators(print_format='matrix'))


'Hamitlonian to have it in groped paulis, call the _check_representation function'
hamiltonian._check_representation(representation)



'Create VQE circuit and execute'

initialstate = Zero(L)
depth=1
var_form = RY(L,depth,initial_state=initialstate)

##param = [-1.29302468,  3.47063906,  0.94745252, -2.13742947]
##vqe_circuit = var_form.construct_circuit(param)

'''
dummy_circ = QuantumCircuit(q)
for i in range(L):
    dummy_circ.h(q[i])
#dummy_circ.h(q[0])
dummy_circ.barrier()
vqe_circuit = dummy_circ
print(vqe_circuit)
'''

##avg, std_dev = hamiltonian.eval( representation, vqe_circuit, backend=backend, run_config={'shots':102400})
##print("Energy of the circuit alone:",avg,std_dev)


max_eval = 10
cobyla = COBYLA(maxiter=max_eval)

'''
vqe = VQE(hamiltonian, var_form, optimizer=cobyla, operator_mode=representation,initial_point=None, aux_operators=None, callback=None)
quantum_instance=QuantumInstance(backend=backend, shots = number_of_shots)
results = vqe.run(quantum_instance)
'''



qmc = QMC(hamiltonian, var_form, optimizer=cobyla, operator_mode=representation, backend=backend, callback=None, nshots=100000)


qmc.run()

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




t=-1.0
J=1.0
problem_type='isi'
L=2
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
print('Eigenvalues is:', ret['wavefunction'])
'Hamitlonian is implicitly converted into matrix in the ED routine, so this is the current stored repr.'
#print(hamiltonian.print_operators(print_format='matrix'))


'Hamitlonian to have it in groped paulis, call the _check_representation function'
hamiltonian._check_representation(representation)



'Visualize the groups'
print_grouped_hamiltonian(hamiltonian)
bases=print_basis(hamiltonian)



'Initialize ancilla register + classical register'
q = QuantumRegister(L, name='q')
qa = QuantumRegister(L, name='qa')
cr = ClassicalRegister(2*L, name='cr')
cv = ClassicalRegister(L, name='cv')
ca = ClassicalRegister(L, name='ca')

system_circuit = QuantumCircuit(qa)


'Create VQE circuit and execute'
'''
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
#dummy_circ.h(q[0])
dummy_circ.barrier()
vqe_circuit = dummy_circ
print(vqe_circuit)

avg, std_dev = hamiltonian.eval( representation, vqe_circuit, backend=backend, run_config={'shots':102400})
print("Energy of the circuit alone:",avg,std_dev)


vqe_alone = QuantumCircuit()
vqe_alone.add_register(cv)

vqe_alone += vqe_circuit
#vqe_alone.measure(q,cv)
#print(vqe_alone)
#result = execute(vqe_alone,backend).result()
#if is_statevector_backend(backend): print(result.get_statevector())
#else: print(result.get_counts())

'''
circuits=hamiltonian.construct_evaluation_circuit( representation, vqe_alone, backend)
result = execute(circuits,backend, shots=100000).result()
avg, std_dev = hamiltonian.evaluate_with_result(representation, circuits, backend, result)
'''


#sys.exit()



'Merge VQE circuit with ancilla and classical register total one'
q_vqe = find_regs_by_name(vqe_circuit, 'q')
system_circuit += vqe_circuit




'Create entangled copy'
for i in range(L):
    system_circuit.cx(q_vqe[i],qa[i])

system_circuit.barrier()

print(system_circuit)




'Generate measurement circuits'

print('#########')
circuits=hamiltonian.construct_evaluation_circuit( representation, system_circuit, backend)
for i in range(len(circuits)):
    print(i)
    circuits[i].barrier()
    circuits[i].add_register(ca)
    circuits[i].measure(qa,ca)
    print(circuits[i])
print('#########')



'Execute'

for i in range(len(circuits)):

    circ = circuits[i]
    print('#', i)
    result = execute(circ,backend, shots=100000).result()
    if is_statevector_backend(backend): print(result.get_statevector())
    else:
        res_raw = result.get_counts()
        print(res_raw)

        res = np.zeros(1<<(2*L),dtype=int)
        for key in res_raw.keys():
            #print(key.replace(" ",""))
            res[int(key.replace(" ",""),2)] = res_raw[key]
        print(res)

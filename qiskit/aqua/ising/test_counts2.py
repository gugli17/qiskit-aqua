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
from qiskit.aqua.utils.backend_utils import is_statevector_backend
from qiskit.visualization import *




n_qubits=2

initialstate = Zero(n_qubits)
print("\nFIRST EXAMPLE\n")

q1 = QuantumRegister(n_qubits, name='q1')
qa = QuantumRegister(n_qubits, name='qa')
c1 = ClassicalRegister(n_qubits, name='c1')
ca = ClassicalRegister(n_qubits, name='ca')

qc1 = QuantumCircuit(qa,q1,ca,c1)
#qc1 = QuantumCircuit(q1,c1)

qc1.h(q1[0])
qc1.h(q1[1])

#qc1.cx(q1[0],qa[0])
#qc1.cx(q1[1],qa[1])

qc1.barrier()

qc_copy = deepcopy(qc1)

qc1.measure(q1,c1)
qc1.measure(qa,ca)

print(qc1)

#backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')

result = execute(qc1,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())




#observable = Operator._measure_pauli_z(result.get_counts(), pauliop[1])
#print(observable)
print("\nSECOND EXAMPLE\n")

meas2 = QuantumCircuit()
meas2.add_register(qa)
meas2.add_register(q1)
meas2.add_register(ca)
meas2.add_register(c1)
meas2.measure(qa, ca)
meas2.measure(q1, c1)

qc2 = qc_copy + meas2

print(qc2)

result = execute(qc2,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())


print("\nTHIRD EXAMPLE (one classical register)\n")

q1 = QuantumRegister(n_qubits, name='q1')
qa = QuantumRegister(n_qubits, name='qa')
cr = ClassicalRegister(2*n_qubits, name='cr')



qc1 = QuantumCircuit(qa,q1,cr)
qc1.h(q1[0])
qc1.h(q1[1])
qc1.barrier()
qc1.cx(q1[0],qa[0])
qc1.cx(q1[1],qa[1])
qc1.barrier()
qc1.measure(0,0)
qc1.measure(1,1)
qc1.measure(2,2)
qc1.measure(3,3)
print(qc1)
result = execute(qc1,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())

pauliop=[1,Pauli(np.ones(2*n_qubits), np.zeros(2*n_qubits))]

print(pauliop[0],pauliop[1])

observable = Operator._measure_pauli_z(result.get_counts(), pauliop[1])
print(observable)


print("\n4 EXAMPLE (concatenating circuit plus one classical register)\n")

q1 = QuantumRegister(n_qubits, name='q1')
qa = QuantumRegister(n_qubits, name='qa')
cr = ClassicalRegister(2*n_qubits, name='cr')

qc1 = QuantumCircuit(q1,cr)

qc1.h(q1[0])
qc1.h(q1[1])
qc1.barrier()

qc1.add_register(qa)
qc1.cx(q1[0],qa[0])
qc1.cx(q1[1],qa[1])
qc1.barrier()
qc1.measure(0,0)
qc1.measure(1,1)
qc1.measure(2,2)
qc1.measure(3,3)

print(qc1)

result = execute(qc1,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())

pauliop=[1,Pauli(np.ones(2*n_qubits), np.zeros(2*n_qubits))]

print(pauliop[0],pauliop[1])

observable = Operator._measure_pauli_z(result.get_counts(), pauliop[1])
print(observable)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
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
from qiskit.aqua.utils import compile_and_run_circuits
from qiskit.aqua.utils.backend_utils import is_statevector_backend
from qiskit.visualization import *




n_qubits=3

initialstate = Zero(n_qubits)


q1 = QuantumRegister(n_qubits, name='q1')
c1 = ClassicalRegister(n_qubits, name='c1')

qc1 = QuantumCircuit(q1,c1)

qc1.x(q1[2])


qc1.barrier()
qc1.measure(q1,c1)

print(qc1)

backend = Aer.get_backend('statevector_simulator')
backend = Aer.get_backend('qasm_simulator')

result = execute(qc1,backend).result()
if is_statevector_backend(backend): print(result.get_statevector())
else: print(result.get_counts())

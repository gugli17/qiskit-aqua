import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute
from qiskit.aqua.utils import find_regs_by_name
from qiskit import BasicAer
import pickle as pkl
from qiskit.providers.aer import Aer, noise
from qiskit import execute
from qiskit.providers.ibmq import IBMQ

#token = '7fef2f33eb4abb6fe936674e5c2e1e3d7863938906ac7364fbfe37821f19d92a496e8ee9c9e84a628131150d6a4a9aade377189e393a3eda922c2728205046e5'
#url = 'https://q-console-api.mybluemix.net/api/Hubs/ibm-q-internal/Groups/zrl/Projects/main'

#IBMQ.enable_account(token, url)

trials = 1
shots = 1024
backend = Aer.get_backend('qasm_simulator')
#backend = IBMQ.get_backend('ibmq_20_tokyo')
print('local_backends available:', Aer.backends())

#device = IBMQ.get_backend('ibmq_20_tokyo')
#properties = device.properties()
#coupling_map = device.configuration().coupling_map

#noise_model = noise.device.basic_device_noise_model(properties)#, gate_times=gate_times)
#basis_gates = noise_model.basis_gates
#noise_model = None
#print(basis_gates)

qb = QuantumRegister(4, 'q')
cb = ClassicalRegister(4, 'c')

base = QuantumCircuit(qb,cb)

base.h(qb[0])
base.h(qb[1])
#base.barrier(qb)
base.cx(qb[0], qb[2])
base.cx(qb[1], qb[3])
#base.barrier(qb)

circuits = []

circ = QuantumCircuit() + base
q = find_regs_by_name(circ, 'q')
c = find_regs_by_name(circ, 'c', qreg=False)
circ.h(q[0])
circ.h(q[1])
#circ.barrier(qb)
circ.measure(q,c)
circuits.append(circ)

circ = QuantumCircuit() + base
q = find_regs_by_name(circ, 'q')
c = find_regs_by_name(circ, 'c', qreg=False)
circ.measure(q,c)
circuits.append(circ)

for i in range(2):
    print(circuits[i].draw())

XX_res = np.zeros((trials,16),dtype=int)
ZZ_res = np.zeros((trials,16),dtype=int)

#print(noise_model)
for trial in range(trials):
    job = execute(circuits, backend=backend, shots=shots)
                       #coupling_map=coupling_map,
                       #basis_gates=basis_gates, noise_model=noise_model)
    result = job.result()

    XX_counts = result.get_counts(circuits[0])
    ZZ_counts = result.get_counts(circuits[1])

    print(XX_counts)
    print(ZZ_counts)

    for key in XX_counts.keys():
        XX_res[trial, int(key,2)] = XX_counts[key]
    for key in ZZ_counts.keys():
        ZZ_res[trial, int(key,2)] = ZZ_counts[key]

    
#np.save('XX_res',XX_res)
#np.save('ZZ_res',ZZ_res)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from copy import deepcopy
import random
import itertools
import numpy as np
from functools import reduce
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



class QMC():
    """
    The Variational Quantum Eigensolver algorithm.

    See https://arxiv.org/abs/1304.3061
    """

    CONFIGURATION = {
        'name': 'QMC',
        'description': 'QMC Inspired Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qmcschema',
            'type': 'object',
            'properties': {
                'operator_mode': {
                    'type': 'string',
                    'default': 'grouped_paulis',
                    'oneOf': [
                        {'enum': ['matrix', 'paulis', 'grouped_paulis']}
                    ]
                },
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
        'depends': [
            {'pluggable_type': 'optimizer',
             'default': {
                     'name': 'L_BFGS_B'
                }
             },
            {'pluggable_type': 'variational_form',
             'default': {
                     'name': 'RY'
                }
             },
        ],
    }

    def __init__(self, operator, var_form, optimizer, backend, operator_mode='grouped_paulis', max_evals_grouped=1, callback=None, nshots=1024):
        """Constructor.

        Args:
            operator (Operator): Qubit operator
            operator_mode (str): operator mode, used for eval of operator
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
            initial_point (numpy.ndarray): optimizer initial point.
            max_evals_grouped (int): max number of evaluations performed simultaneously
            aux_operators (list of Operator): Auxiliary operators to be evaluated at each eigenvalue
            callback (Callable): a callback that can access the intermediate data during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard devation.
        """
        #self.validate(locals())
        #super().__init__(var_form=var_form,
        #                 optimizer=optimizer,
        #                 cost_fn=self._energy_evaluation,
        #                 initial_point=initial_point)
        #self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        self._optimizer = optimizer
        #if initial_point is None:
        #    self._initial_point = var_form.preferred_init_points
        self._operator = operator
        self._var_form = var_form
        self._operator_mode = operator_mode
        self._backend = backend
        self._L = operator.num_qubits
        self._N = 2*self._L
        self._nshots = nshots

        c=1.0/np.sqrt(2)
        self.postrotation = {
                "X": [[c+0.j,  c+0.j],[ c+0.j, -c+0.j]],
                "Y": [[c+0.j,     c*1.j],[c+0.j,   -c*1.j]],
                "Z": [[1 , 0.],[0 ,1]]
                }


    def run(self):


        '''
        energy, params, circuit = self._optimize_starting_circuit()
        print( energy)
        print("\n----VQE circuit------\n",circuit)
        self._optimal_vqe_circ = circuit

        '''
        #param = [-1.29302468,  3.47063906,  0.94745252, -2.13742947]
        param = [ 1.19399161 , 1.4833124  , 0.88424459,  0.7803776 ,  1.08447047 ,-0.30487949]
        self._optimal_vqe_circ = self._var_form.construct_circuit(param)
        backendsv = Aer.get_backend('statevector_simulator')
        avg, std_dev = self._operator.eval( self._operator_mode, self._optimal_vqe_circ, backend=backendsv, run_config={'shots':self._nshots})
        print("Energy of the circuit alone:",avg,std_dev)

        avg, std_dev = self._operator.eval( self._operator_mode, self._optimal_vqe_circ, backend=self._backend, run_config={'shots':self._nshots})
        print("Energy of the circuit alone:",avg,std_dev)



        self._construct_entcopy_circuit()
        print("\n----Extended circuit------\n",self.circuit)
        self._construct_measurements_circuits()
        psi2 = self._eval_circuits()

        basis_set = self._construct_basis()
        print(basis_set)
        psi2 = np.reshape(psi2,(len(self.m_circuits),1<<self._N,1))
        #print(psi2)
        psi2_reduced=self._compute_allreduced(psi2,basis_set)
        #print(psi2_reduced)


        counts=self.from_array_to_counts(psi2_reduced)
        for i in range(len(basis_set)):
            print(basis_set[i], counts[i])


        return 0


    def _optimize_starting_circuit(self):


        vqe = VQE(self._operator, self._var_form, optimizer=self._optimizer, operator_mode=self._operator_mode, initial_point=None, aux_operators=None, callback=None)
        number_of_shots = self._nshots
        quantum_instance=QuantumInstance(backend=self._backend, shots = number_of_shots)
        results = vqe.run(quantum_instance)
        print('The computed ground state energy is: {:.12f}'.format(results['energy']))
        print("Parameters: {}".format(results['opt_params']))

        return results['energy'], results['opt_params'], vqe.get_optimal_circuit()



    def _construct_entcopy_circuit(self):

        qa = QuantumRegister(self._L, name='qa')
        ca = ClassicalRegister(self._L, name='ca')

        system_circuit = QuantumCircuit(qa)
        q_vqe = find_regs_by_name(self._optimal_vqe_circ, 'q')
        system_circuit += self._optimal_vqe_circ


        for i in range(self._L):
            system_circuit.cx(q_vqe[i],qa[i])

        system_circuit.barrier()


        self.circuit=system_circuit

        return


    def _construct_measurements_circuits(self):

        ca = ClassicalRegister(self._L, name='ca')
        qa = find_regs_by_name(self.circuit, 'qa')
        circuits=self._operator.construct_evaluation_circuit( self._operator_mode, self.circuit, self._backend)
        for i in range(len(circuits)):
            #print("\nprinting circ. with meas..",i,"\n")
            circuits[i].barrier()
            circuits[i].add_register(ca)
            circuits[i].measure(qa,ca)
            print(circuits[i])


        self.m_circuits=circuits

        return


    def _eval_circuits(self):

        psi2=[]
        for i in range(len(self.m_circuits)):

            circ = self.m_circuits[i]
            #print('#', i)
            result = execute(circ,self._backend, shots=self._nshots).result()

            res_raw = result.get_counts()
            #print(res_raw)

            res = np.zeros(1<<(2*self._L),dtype=int)
            for key in res_raw.keys():
                #print(key.replace(" ",""))
                res[int(key.replace(" ",""),2)] = res_raw[key]
            #print(res)
            res = np.squeeze(res) / self._nshots
            psi2.append(res)

        return psi2


    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            float or list of float: energy of the hamiltonian of each parameter.
        """
        return 0


    def _construct_basis(self):

        if self._operator._grouped_paulis is not None:
            n_qubits=self._operator.num_qubits
            print('\nbasis of grouped paulis')
            baseslist=[]
            for idx, tpb_set in enumerate(self._operator.grouped_paulis):
                print('group:',idx)
                list=[]
                for qubit_idx in range(n_qubits-1,-1,-1):     #important for the qiskit convention!
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
        elif self._operator._paulis is not None:
            n_qubits=self._operator.num_qubits
            print('\nbasis of paulis')
            baseslist=[]
            for idx, pauli in enumerate(self._operator.paulis):
                print('group:',idx)
                list=[]
                for qubit_idx in range(n_qubits-1,-1,-1):  #important for the qiskit convention!
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
            raise QiskitError('Matrix representation not supported.')


        return baseslist


    def _compute_allreduced(self,psifull2,basis_set):
        '''
        TODO: must remove any exponentially scaling variables / loops
        '''
        L=self._L
        D = 1<< self._L
        psi2          = []

        #print(st)

        for b in range(len(basis_set)):
            tmp=[]
            norm=0
            for comp in range(D):
                st= np.binary_repr(comp, width=L)
                state=np.array(list(st),dtype=int)
                psi_b2_state= self._compute_reduced(state, psifull2[b], basis_set[b])
                #print(psi_b2_state.real)
                norm+=psi_b2_state[0].real
                tmp.append(psi_b2_state[0].real)
            print(norm)
            tmp[:] = [x / norm for x in tmp]  #####<-added to renormalize P
            psi2.append(tmp)

        return psi2



    def _compute_reduced(self,targetstate,psi2,basis):

        idx=0
        L=self._L
        p=0
        print(targetstate)
        # notice in np array, pos 0 is most significant digit!
        for i in range(L):
            #idx+=targetstate[(L-1)-i]*(1<<i)
            idx+=targetstate[i]*(1<<i)

        print(idx)
        period = (1<<L)

        #print("basis inside",basis)
        ctr=0
        for i in range(len(basis)):
            if basis[i]=='Z': ctr+=1

        signmat=self.signmat(basis)*np.power(np.sqrt(2),L-ctr)
        #print(signmat)
        ###########
        #if basis==['Z', 'Z']:
        #    signmat = np.ones((period,period))
        ###########
        for i in range(period):
            j = idx + i*period
            state_full= np.binary_repr(j, width=self._N)
            idx_psi=0
            v=state_full[L:]
            for i in range(L):
                idx_psi+=int(v[(L-1)-i])*(1<<i)
                #idx_psi+=int(v[i])*(1<<i)
            v=state_full[:L]
            idx_a=0
            for i in range(L):
                idx_a+=int(v[(L-1)-i])*(1<<i)
                #idx_a+=int(v[i])*(1<<i)
            #print(j, state_full, state_full[:N], idx_a, state_full[N:], idx_psi, psi2[j])


            sign=signmat[idx_psi,idx_a]

            p+= sign*np.sqrt(psi2[j])




        p=p*np.conj(p)

        return p


    def signmat(self,basis):
        L=self._L
        list = []
        #for k in range(L):
        for k in range(L-1,-1,-1):
            for key in self.postrotation.keys():
                if (basis[k] == key):
                    list.append(self.postrotation[key])

        return reduce(np.kron,list)


    def from_array_to_counts(self,vect):
        countslist=[]
        D = 1<< self._L
        for n in range(len(self.m_circuits)):
            array=vect[n]
            dict={}
            for i in range(D):
                key=np.binary_repr(i, width=self._L)
                dict[key]=int(round(array[i]*self._nshots))

            countslist.append(dict)

            sum=0
            for key in dict.keys():
                sum += dict[key]
            print("sum",sum, self._nshots)

        #print(countslist)
        return countslist

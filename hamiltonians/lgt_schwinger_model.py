from qiskit.opflow import PauliOp
from qiskit.quantum_info.operators import Pauli
import warnings

import numpy as np
from typing import Optional, Union, List


class LGTSchwingerModel:
    """

    """

    def __init__(self,
                 mass: float,
                 couple: float,
                 num_sites: int,
                 spacing: float,
                 left_gauge: Optional[float] = 0):
        """
        Args:
            mass: particle mass
            couple: coupling constant (e.g., -e for QED)
            num_sites: number of lattice sites
            spacing: lattice constant
        """
        if num_sites % 2 != 0:
            warnings.warn(f'number of sites is {num_sites} but should be even for staggered fermions!')

        self._mass = mass
        self._couple = couple
        self._num_sites = num_sites
        self._spacing = spacing
        self._left_gauge = left_gauge


    def _build_hopping_ops(self):
        """
        Constructs all hopping terms of the Hamiltonian after qubit-mapping.
        These are (sigma^+ sigma^_ + h.c.) terms which simplify to (YY + XX) terms
        and are split into mutually non-commuting even and odd sums of Paulis within which all terms
        commute.
        """
        coeff = 1./(4*self._spacing)
        op_even = []
        op_odd = []
        for j in range(0, self._num_sites-1, 2):
            op_even.append([coeff, Pauli((self._num_sites - 1 - j - 1)*"I" + "X" + "X" + j*"I")])
            op_even.append([coeff, Pauli((self._num_sites - 1 - j - 1)*"I" + "Y" + "Y" + j*"I")])
        for j in range(1, self._num_sites-1, 2):
            op_odd.append([coeff, Pauli((self._num_sites - 1 - j - 1)*"I" + "X" + "X" + j*"I")])
            op_odd.append([coeff, Pauli((self._num_sites - 1 - j - 1)*"I" + "Y" + "Y" + j*"I")])

        return [op_even, op_odd]


    def _build_z_ops(self):
        """
        Constructs all Pauli operators constituting the mass term as well as the gauge-field term of the
        Hamiltonian with staggered fermions.
        These operators consist of only I and Z.
        """
        # NOTE: sum is shifted from {1,...,N} to {0,...N-1}. This changes the mass-term's sign at each
        # site because of the (-1)^j coefficient.. do I need to change this factor to (-1)^{j+1}?

        op_z = []

        coeff_mass = 0.5*self._mass # *self._spacing # NOTE
        coeff_couple = 0.5*self._couple*self._couple*self._spacing # *self._spacing # NOTE
        Nmod2 = (self._num_sites % 2)
        coeff_const = (-coeff_mass*Nmod2
                       + coeff_couple*((self._num_sites - 1)*self._left_gauge*self._left_gauge
                                       + (0.125 - 0.5*self._left_gauge)*(self._num_sites - Nmod2)))

        for j in range(self._num_sites - 1):
            op_z_j = []
            # mass terms
            op_z_j.append([coeff_mass*(1 if j % 2 == 0 else -1),
                           Pauli((self._num_sites - 1 - j)*'I' + 'Z' + j*'I')])

            # NOTE coeff_j = (self._left_gauge - 0.5*(j % 2))
            coeff_j = (self._left_gauge + 0.5*((j+1) % 2))

            # gauge terms from Gauss' law
            for l in range(j + 1): # need +1 as otherwise sum stops at N-3 instead of N-2
                op_z_j.append([coeff_j*coeff_couple, Pauli((self._num_sites - 1 - l)*'I' + 'Z' + l*'I')])
                pauli1 = Pauli((self._num_sites - 1 - l)*'I' + 'Z' + l*'I')

                for ll in range(j + 1):
                    if l == ll:
                        coeff_const += 0.25*coeff_couple
                    else:
                        pauli2 = Pauli((self._num_sites - 1 - ll)*'I' + 'Z' + ll*'I')
                        op_z_j.append([0.25*coeff_couple, pauli1.dot(pauli2)])

            op_z.append(op_z_j)

        op_z.append([[coeff_mass*(1 if (self._num_sites - 1) % 2 == 0 else -1),
                     Pauli('Z' + (self._num_sites - 1)*'I')]]) # j = N mass term

        # const term last, so it can be easily discarded
        # op_z.append([[coeff_const, Pauli(self._num_sites*'I')]])

        return op_z


    def as_pauli_op(self):
        """
        Returns the Hamiltonian as a list of operators. Each operator is a sum of Pauli operators.
        The operators are grouped such that within each sum, all Paulis commute but the elements of the
        list (i.e., Pauli sums) do not commute mutually.
        """
        pauli_op = []
        # pauli_op.extend(self._build_hopping_ops())
        # pauli_op.extend(self._build_z_ops())

        hopping_ops = self._build_hopping_ops()
        coupling_ops = self._build_z_ops()

        for op_list in hopping_ops:
            summed_op = 0
            for op in op_list:
                summed_op += PauliOp(op[1], coeff=op[0])
            pauli_op.append(summed_op)

        summed_op = 0 # all operators returned by _build_staggered_z_ops() commute
        for op_list in coupling_ops:
            for op in op_list:
                summed_op += PauliOp(op[1], coeff=op[0])
        pauli_op.append(summed_op)

        return pauli_op

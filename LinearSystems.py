import math

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

VALID_SYSTEM_TYPES = [
    'gauss_elimination',
    'gauss_elimination_rounding'
]


class LinearSystemsFactory:
    def __init__(self):
        self._linear_systems = {}
        for linear_system in LinearSystemsBase.__subclasses__():
            linear_system_obj = linear_system()
            self._linear_systems[linear_system_obj._get_system_type()] = linear_system_obj

    def get_linear_system(self, system_type):
        if system_type not in VALID_SYSTEM_TYPES:
            raise ValueError('The linear system type supplied is not valid.'
                             'You Must enter linear system Type')
        return self._linear_systems[system_type]

class LinearSystemsHelper:
    @staticmethod
    def pivot(a, pivot_index):
        pivot_row = a[pivot_index].copy()
        max_row = pivot_row.copy()
        max_row_index = pivot_index
        for i in range(pivot_index + 1, a.shape[0]):
            if math.fabs(a[i, pivot_index]) > math.fabs(max_row[pivot_index]):
                max_row = a[i].copy()
                max_row_index = i
        if pivot_index != max_row_index:
            print(f'Pivoting R{pivot_index + 1}  <--> R{max_row_index + 1}')
            a[pivot_index] = max_row
            a[max_row_index] = pivot_row
            print(f'Output:\n{a}\n\n')

class LinearSystemsBase(ABC):
    @abstractmethod
    def _get_system_type(self):
        pass

    @abstractmethod
    def solve(self, **kwargs):
        pass

class GaussElimination(LinearSystemsBase):
    def _get_system_type(self):
        return 'gauss_elimination'

    def solve(self, **kwargs):
        print('Starting Gauss Elimination with rounding')
        self._a = kwargs.get('a')
        self._b = kwargs.get('b')
        self._pivoting = kwargs.get('pivoting')
        if self._b is not None:
            self._a = np.concatenate((self._a, self._b.T), axis=1)
        print(f'Augmented Matrix: \n{self._a}\n\n')
        self.forward_elimination()
        self.backward_substitution()

    def forward_elimination(self):
        print('Performing forward elimination:\n')
        a = self._a.astype(float)
        num_rows = a.shape[0]
        for i in range(num_rows):
            if self._pivoting == True:
                LinearSystemsHelper.pivot(a, i)
            for j in range(i + 1, num_rows):
                if a[j, i] != 0:
                    print(f'R{j + 1} -> R{j + 1} - {a[j, i]}/{a[i, i]}*R{i + 1}')
                    a[j] -= a[j, i] / a[i, i] *a[i]
                    a[j, i] = 0
                    print(f'Output:\n{a}\n\n')
        print(f'Final REF form: \n {a}')
        self._a = a

    def backward_substitution(self):
        a = self._a
        print('\n\nPerforming backward substitution:\n')
        num_vars = a.shape[1] - 1
        # print(num_vars)
        num_rows = a.shape[0]
        # print(num_rows)
        if num_vars <= num_rows:
            for i in range(num_rows):
                if sum(a[i, :-1]) <= 1.0e-12 and a[i, num_vars] != 0:
                    print('Given system of equations is inconsistent')
                    return

            var_names = []
            var_counter = num_vars
            var_values = np.ones(num_vars + 1)
            # print(var_values)
            for j in range(num_vars - 1, -1, -1):
                if len(var_names) == 0:
                    row = a[j]
                    row_op = row * var_values
                    rhs = sum(row_op) - row_op[j] - row_op[-1]
                    var_val = (row_op[-1] - rhs) / row_op[j]
                    var_val = var_val
                    print(f'From R{j + 1}, we get X{j + 1} = {var_val}')
                    var_values[j] = var_val
                    var_names.append(f'X{j + 1}')
                else:
                    row = a[j]
                    row_op = row * var_values
                    rhs = sum(row_op) - row_op[j] - row_op[-1]
                    var_val = (row_op[-1] - rhs) / row_op[j]
                    var_val = var_val
                    var_list = list(var_values)
                    var_list.reverse()
                    var_list = var_list[1:len(var_names) + 1]
                    var_str = [x + ' = ' + str(y) for x, y in zip(var_names, var_list)]
                    print(f'Substituting {var_str} in R{j + 1}, we get X{j + 1} = {var_val}')
                    var_values[j] = var_val
                    var_names.append(f'X{j + 1}')
        elif num_vars > num_rows:
            print('Given system of equations have infinitely many solutions')

class GaussEliminationRounding(LinearSystemsBase):

    round_to_n = staticmethod(lambda a, n: np.array([x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)) for x in a]) \
        if isinstance(a, np.ndarray) else a if a == 0 else round(a, -int(math.floor(math.log10(abs(a)))) + (n - 1)))

    def _get_system_type(self):
        return 'gauss_elimination_rounding'

    def solve(self, **kwargs):
        print('Starting Gauss Elimination with rounding')
        self._a = kwargs.get('a')
        self._b = kwargs.get('b')
        self._pivoting = kwargs.get('pivoting')
        self._rounding_digit = kwargs.get('rounding_digit')
        if self._b is not None:
            self._a = np.concatenate((self._a, self._b.T), axis=1)
        print(f'Augmented Matrix: \n{self._a}\n\n')
        self.forward_elimination()
        self.backward_substitution()

    def forward_elimination(self):
        print('Performing forward elimination:\n')
        a = self._a.astype(float)
        num_rows = a.shape[0]
        for i in range(num_rows):
            if self._pivoting == True:
                LinearSystemsHelper.pivot(a, i)
            for j in range(i + 1, num_rows):
                if a[j, i] != 0:
                    print(f'R{j + 1} -> R{j + 1} - {a[j, i]}/{a[i, i]}*R{i + 1}')
                    a[j] -= self.round_to_n(self.round_to_n(a[j, i] / a[i, i], self._rounding_digit) * self.round_to_n(a[i], self._rounding_digit), self._rounding_digit)
                    a[j] = self.round_to_n(a[j], self._rounding_digit)
                    a[j, i] = 0
                    print(f'Output:\n{a}\n\n')
        print(f'Final REF form: \n {a}')
        self._a = a

    def backward_substitution(self):
        a = self._a

        print('\n\nPerforming backward substitution:\n')
        num_vars = a.shape[1] - 1
        # print(num_vars)
        num_rows = a.shape[0]
        # print(num_rows)
        if num_vars <= num_rows:
            for i in range(num_rows):
                if sum(a[i, :-1]) == 0 and a[i, num_vars] != 0:
                    print('Given system of equations is inconsistent')
                    return

            var_names = []
            var_counter = num_vars
            var_values = np.ones(num_vars + 1)
            # print(var_values)
            for j in range(num_vars - 1, -1, -1):
                if len(var_names) == 0:
                    row = a[j]
                    row_op = self.round_to_n(row * var_values, self._rounding_digit)
                    rhs = self.round_to_n(sum(row_op) - row_op[j] - row_op[-1], self._rounding_digit)
                    var_val = (row_op[-1] - rhs) / row_op[j]
                    var_val = self.round_to_n(var_val, self._rounding_digit)
                    print(f'From R{j + 1}, we get X{j + 1} = {var_val}')
                    var_values[j] = var_val
                    var_names.append(f'X{j + 1}')
                else:
                    row = a[j]
                    row_op = self.round_to_n(row * var_values, self._rounding_digit)
                    rhs = self.round_to_n(sum(row_op) - row_op[j] - row_op[-1], self._rounding_digit)
                    var_val = (row_op[-1] - rhs) / row_op[j]
                    var_val = self.round_to_n(var_val, self._rounding_digit)
                    var_list = list(var_values)
                    var_list.reverse()
                    var_list = var_list[1:len(var_names) + 1]
                    var_str = [x + ' = ' + str(y) for x, y in zip(var_names, var_list)]
                    print(f'Substituting {var_str} in R{j + 1}, we get X{j + 1} = {var_val}')
                    var_values[j] = var_val
                    var_names.append(f'X{j + 1}')
        elif num_vars > num_rows:
            print('Given system of equations have infinitely many solutions')


if __name__ == '__main__':
    a = np.array([[3,2,1], [2,1,1], [6,2,4] ])
    b =np.array([[3,0,6]])
    lsf = LinearSystemsFactory()
    lsf.get_linear_system(system_type='gauss_elimination').solve(a=a, b = b, pivoting = True)
    lsf.get_linear_system(system_type='gauss_elimination_rounding').solve(a=a, b=b, rounding_digit = 3)
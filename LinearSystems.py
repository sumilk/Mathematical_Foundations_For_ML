import math

import pandas as pd
import numpy as np
from sympy import Matrix
from abc import ABC, abstractmethod

VALID_SYSTEM_TYPES = [
    'gauss_elimination',
    'gauss_elimination_rounding',
    'condition_number',
    'linear_independence_checker',
    'basis_finder',
    'gauss_seidel',
    'gauss_jacobi',
    'rayleigh_quotient'
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
                             'You Must enter a Valid linear system Type')
        return self._linear_systems[system_type]

class LinearSystemsHelper:
    round_to_n = staticmethod(lambda a, n: np.array([x if x == 0 else round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1)) for x in a]) \
            if isinstance(a, np.ndarray) else a if a == 0 else round(a, -int(math.floor(math.log10(abs(a)))) + (n - 1)))
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

    @staticmethod
    def check_for_diagonal_dominance(a):
        diag = np.diag(np.abs(a))
        #print(diag)

        # Find row sum without diagonal
        off_diag = np.sum(np.abs(a), axis=1) - diag

        if np.all(diag > off_diag):
            print('Matrix is diagonally dominant\n')
        else:
            print('Matrix is NOT diagonally dominant\n')

    @staticmethod
    def generate_random_matrix(size, diag_dom):
        eps = 0.1
        A = LinearSystemsHelper.gen_diag_matrix(size) if diag_dom \
            else np.random.randint(-10, 10, size) / (np.random.rand() + eps)
        # print(f'Input Matrix A:\n\n{A}\n\n')
        b = np.random.randint(-10, 10, size[0]) / (np.random.rand() + eps)
        # print(f'b:\n\n{b}\n\n')
        return A,b

    @staticmethod
    def gen_diag_matrix(size):
        eps = 0.1
        matrix = np.random.randint(-10, 10, (size)) / (np.random.rand() + eps)
        # print(matrix)
        sign_matrix = np.where(np.sign(matrix) == 0, 1, np.sign(matrix))

        np.fill_diagonal(matrix, np.sum(np.abs(matrix), axis=1)
                         + np.diag(np.abs(matrix)) + eps)  # added constant to ensure strict diagonal dominance
        return np.abs(matrix) * sign_matrix


# noinspection PyAbstractClass
class LinearSystemsBase(ABC):
    @abstractmethod
    def _get_system_type(self):
        pass

    @abstractmethod
    def solve(self, **kwargs):
        pass


class RayleighQuotient(LinearSystemsBase):
    def _get_system_type(self):
        return 'rayleigh_quotient'


    def solve(self, **kwargs):
        print('\nPerforming Rayleigh Quotient iterations to find dominant eigenvalue\n')
        self._A = kwargs.get('A')
        self.size = kwargs.get('size') if self._A is None else self._A.shape
        self.rounding_digit = kwargs.get('rounding_digit') if kwargs.get('rounding_digit') is not None else 10
        if self._A is None:
            if kwargs.get('randomize'):
                self._A, self._b = LinearSystemsHelper.generate_random_matrix(self.size, kwargs.get('diag_dom'))
            else:
                raise ValueError('Please provide input matrix A or set randomize flag to True')

        self._A = np.array([LinearSystemsHelper.round_to_n(x, self.rounding_digit) for x in self._A])

        self.x_init = kwargs.get('x_init') if kwargs.get('x_init') is not None else np.ones(self._A.shape[1])
        self.n_iterations = kwargs.get('n_iterations') if kwargs.get('n_iterations') is not None else 10
        self.print_iterations = kwargs.get('print_iterations')
        self.plot_iterations = kwargs.get('plot_iterations')

        self.tolerance = kwargs.get('tolerance') if kwargs.get('tolerance') is not None else 0.0001

        self.perform_Rayleigh_Iterations()

    def perform_Rayleigh_Iterations(self):
        print(f'\nInput Matrix \'A\':\n{self._A}\n')
        print(f'\nInitial value x: {self.x_init}')
        x_norm = []
        x = LinearSystemsHelper.round_to_n(np.array(self.x_init), self.rounding_digit)
        # epsilon = 0.001
        converged = False
        #q=0
        for i in range(self.n_iterations):
            print(f'\n\nIteration: {i+1}\n')
            print(f'x = {x}\n')
            y = LinearSystemsHelper.round_to_n(self._A@x, self.rounding_digit)
            print(f'y = Ax = {y}\n')

            m_0 = LinearSystemsHelper.round_to_n(x@x, self.rounding_digit)
            print(f'm\u2080 = x\u1D40x = {m_0}')

            m_1 = LinearSystemsHelper.round_to_n(x@y, self.rounding_digit)
            print(f'm\u2081 = x\u1D40y = {m_1}')

            m_2 = LinearSystemsHelper.round_to_n(y@y, self.rounding_digit)
            print(f'm\u2082 = y\u1D40y = {m_2}')

            y_norm = LinearSystemsHelper.round_to_n(y/max(y), self.rounding_digit)
            print(f'y_norm = y/max(y) = {y_norm}')

            q = LinearSystemsHelper.round_to_n(m_1/m_0, self.rounding_digit)
            print(f'q = m\u2081/m\u2080 = {q}')

            x = y_norm
            delta = LinearSystemsHelper.round_to_n(np.sqrt(np.abs(m_2/m_0 - q**2)), self.rounding_digit)

            print(f'delta = sqrt(m\u2082/m\u2080 - q\u00B2) =  {delta}')


            if self.plot_iterations:
                x_norm.append(delta)

            if delta < self.tolerance:
                converged = True
                print('Converged!')
                print(f'\nEigenvalue estimate using Rayleigh Quotient: {q}')
                break

        if not converged:
            print(f'Not converged after {self.n_iterations} iterations')

        if self.plot_iterations:
            plt.plot(x_norm)
            plt.xlabel('Iterations')
            plt.ylabel('Difference')


class GaussSeidel(LinearSystemsBase):
    def _get_system_type(self):
        return 'gauss_seidel'

    def solve(self, **kwargs):
        print('\nPerforming Gauss Seidel Iterations\n')
        self._A = kwargs.get('A')
        self._b = kwargs.get('b')
        self.size = kwargs.get('size') if self._A is None else self._A.shape
        self.rounding_digit = kwargs.get('rounding_digit') if kwargs.get('rounding_digit') is not None else 10
        if self._A is None:
            if kwargs.get('randomize'):
                self._A, self._b = LinearSystemsHelper.generate_random_matrix(self.size, kwargs.get('diag_dom'))
            else:
                raise ValueError('Please provide input matrix A and b or set randomize flag to True')

        self._A = np.array([LinearSystemsHelper.round_to_n(x, self.rounding_digit) for x in self._A])
        self._b = LinearSystemsHelper.round_to_n(self._b, self.rounding_digit)

        self.x_init = kwargs.get('x_init') if kwargs.get('x_init') is not None else np.zeros(self._A.shape[1])
        self.n_iterations = kwargs.get('n_iterations') if kwargs.get('n_iterations') is not None else 10
        self.print_iterations = kwargs.get('print_iterations')
        self.plot_iterations = kwargs.get('plot_iterations')
        self.tolerance = kwargs.get('tolerance') if kwargs.get('tolerance') is not None else 0.001

        self.generate_Gauss_Seidel_Iteration()
        self.perform_Gauss_Seidel_Iterations()

    def generate_Gauss_Seidel_Iteration(self, x=None):
        if x is None:
            A = self._A
            b = self._b
            A = A.astype(float)
            if b is not None:
                b = b.astype(float)
            print(f'Input Matrix \'A\':\n{A}\n')
            LinearSystemsHelper.check_for_diagonal_dominance(A)
            print('\nReducing diagonal elements to 1\n')
            for i in range(0, A.shape[0]):
                div = A[i, i]
                A[i] = LinearSystemsHelper.round_to_n(A[i] / div, self.rounding_digit)
                if b is not None:
                    b[i] = LinearSystemsHelper.round_to_n(b[i] / div, self.rounding_digit)
            print(f'A: \n{A}\n')
            if b is not None:
                print(f'b: \n{b}\n')
            self.I = np.eye(A.shape[0])
            self.L = np.tril(A) - self.I
            self.U = np.triu(A) - self.I
            print('A = I + L + U, we get\n')
            print(f'I: \n{self.I}\n\nL:\n{self.L}\n\nU:\n{self.U}')
            self.iteration_matrix = np.linalg.inv(self.I + self.L) @ self.U
            self.iteration_matrix = np.array([LinearSystemsHelper.round_to_n(x, self.rounding_digit) for x in self.iteration_matrix ])
            print(f'\nIteration Matrix (I+L)\u207B\u00B9U: \n{self.iteration_matrix}')
            # print(np.linalg.inv(I+L)@U)
            self.norm_one = np.linalg.norm(self.iteration_matrix, 1)
            self.norm_inf = np.linalg.norm(self.iteration_matrix, np.inf)
            self.norm_fro = np.linalg.norm(self.iteration_matrix, 'fro')
            print(f'\nnorm_1 : {self.norm_one}\nnorm_inf : {self.norm_inf}\nnorm_frobenius : {self.norm_fro}\n')
            self._A = A
            self._b = b
        else:
            x = LinearSystemsHelper.round_to_n(np.linalg.inv(self.I + self.L) @ self._b, self.rounding_digit)  \
                - LinearSystemsHelper.round_to_n(self.iteration_matrix @ x, self.rounding_digit)
            x = LinearSystemsHelper.round_to_n(x, self.rounding_digit)
        return x

    def perform_Gauss_Seidel_Iterations(self):
        print(f'\n\nInitial value x: {self.x_init}')
        x_norm = []
        x_prev = np.array(self.x_init)
        #epsilon = 0.001
        converged = False
        for i in range(self.n_iterations):
            x = list(self.generate_Gauss_Seidel_Iteration(x_prev))
            diff_x = np.linalg.norm(np.array(x) - x_prev, 2)
            # print(diff_x)
            x_prev = np.array(x)
            if self.print_iterations:
                print(f'Iteration - {i + 1}, x_{i+1}  = (I+L)\u207B\u00B9*b - (I+L)\u207B\u00B9U*x_{i} = {x}')
            if self.plot_iterations:
                x_norm.append(diff_x)

            if diff_x < self.tolerance:
                converged = True
                print('Converged!')
                break

        if not converged:
            print(f'Not converged after {self.n_iterations} iterations')

        if self.plot_iterations:
            plt.plot(x_norm)
            plt.xlabel('Iterations')
            plt.ylabel('Difference')

class GaussJacobi(LinearSystemsBase):
    def _get_system_type(self):
        return 'gauss_jacobi'

    def solve(self, **kwargs):
        print('\nPerforming Gauss Jacobi Iterations\n')
        self._A = kwargs.get('A')
        self._b = kwargs.get('b')
        self.size = kwargs.get('size') if self._A is None else self._A.shape
        self.rounding_digit = kwargs.get('rounding_digit') if kwargs.get('rounding_digit') is not None else 10
        if self._A is None:
            if kwargs.get('randomize'):
                self._A, self._b = LinearSystemsHelper.generate_random_matrix(self.size, kwargs.get('diag_dom'))
            else:
                raise ValueError('Please provide input matrix A and b or set randomize flag to True')

        self._A = np.array([LinearSystemsHelper.round_to_n(x, self.rounding_digit) for x in self._A])
        self._b = LinearSystemsHelper.round_to_n(self._b, self.rounding_digit)

        self.x_init = kwargs.get('x_init') if kwargs.get('x_init') is not None else np.zeros(self._A.shape[1])
        self.n_iterations = kwargs.get('n_iterations') if kwargs.get('n_iterations') is not None else 10
        self.print_iterations = kwargs.get('print_iterations')
        self.plot_iterations = kwargs.get('plot_iterations')
        self.tolerance = kwargs.get('tolerance') if kwargs.get('tolerance') is not None else 0.001

        self.generate_Gauss_Jacobi_Iteration()
        self.perform_Gauss_Jacobi_Iterations()

    def generate_Gauss_Jacobi_Iteration(self, x=None):
        if x is None:
            A = self._A
            b = self._b
            A = A.astype(float)
            if b is not None:
                b = b.astype(float)
            print(f'Input Matrix \'A\':\n{A}\n')
            LinearSystemsHelper.check_for_diagonal_dominance(A)
            print('\nReducing diagonal elements to 1\n')
            for i in range(0, A.shape[0]):
                div = A[i, i]
                A[i] = LinearSystemsHelper.round_to_n(A[i] / div, self.rounding_digit)
                if b is not None:
                    b[i] = LinearSystemsHelper.round_to_n(b[i] / div, self.rounding_digit)
            print(f'A: \n{A}\n')
            if b is not None:
                print(f'b: \n{b}\n')
            self.I = np.eye(A.shape[0])
            self.L = np.tril(A) - self.I
            self.U = np.triu(A) - self.I
            print('A = I + L + U, we get\n')
            print(f'I: \n{self.I}\n\nL:\n{self.L}\n\nU:\n{self.U}')
            self.iteration_matrix = self.L + self.U
            self.iteration_matrix = np.array([LinearSystemsHelper.round_to_n(x, self.rounding_digit) for x in self.iteration_matrix ])
            print(f'\nIteration Matrix (L+U): \n{self.iteration_matrix}')
            # print(np.linalg.inv(I+L)@U)
            self.norm_one = np.linalg.norm(self.iteration_matrix, 1)
            self.norm_inf = np.linalg.norm(self.iteration_matrix, np.inf)
            self.norm_fro = np.linalg.norm(self.iteration_matrix, 'fro')
            print(f'\nnorm_1 : {self.norm_one}\nnorm_inf : {self.norm_inf}\nnorm_frobenius : {self.norm_fro}\n')
            self._A = A
            self._b = b
        else:
            x = LinearSystemsHelper.round_to_n(self._b, self.rounding_digit)  \
                - LinearSystemsHelper.round_to_n(self.iteration_matrix @ x, self.rounding_digit)
            x = LinearSystemsHelper.round_to_n(x, self.rounding_digit)
        return x

    def perform_Gauss_Jacobi_Iterations(self):
        print(f'\n\nInitial value x: {self.x_init}')
        x_norm = []
        x_prev = np.array(self.x_init)
        #epsilon = 0.001
        converged = False
        for i in range(self.n_iterations):
            x = list(self.generate_Gauss_Jacobi_Iteration(x_prev))
            diff_x = np.linalg.norm(np.array(x) - x_prev, 2)
            # print(diff_x)
            x_prev = np.array(x)
            if self.print_iterations:
                print(f'Iteration - {i + 1}, x_{i+1}  = b - (L+U)*x_{i} = {x}')
            if self.plot_iterations:
                x_norm.append(diff_x)

            if diff_x < self.tolerance:
                converged = True
                print('Converged!')
                break

        if not converged:
            print(f'Not converged after {self.n_iterations} iterations')

        if self.plot_iterations:
            plt.plot(x_norm)
            plt.xlabel('Iterations')
            plt.ylabel('Difference')


class ConditionNumber(LinearSystemsBase):
    def _get_system_type(self):
        return 'condition_number'

    def solve(self, **kwargs):
        print('Finding condition number')
        self._a = kwargs.get('a')
        if self._a is None:
            raise ValueError('Input Matrix is Missing')
        print(f'Input Matrix:\n\n{self._a}\n\n')
        cond_one = np.linalg.norm(self._a,1)*np.linalg.norm(np.linalg.inv(self._a),1)
        cond_inf = np.linalg.norm(self._a, np.inf) * np.linalg.norm(np.linalg.inv(self._a), np.inf)
        print(f'Condition Number(1): {cond_one}\nCondition Number(inf): {cond_inf}')
        return cond_one, cond_inf


class LinearIndependenceChecker(LinearSystemsBase):
    def _get_system_type(self):
        return 'linear_independence_checker'
    def solve(self, **kwargs):
        linear_Independent = False
        try:
            matrix = np.array(list(kwargs.values()))
            #print(matrix.shape)
            #print(np.linalg.matrix_rank(matrix))
            if matrix.shape[0] == np.linalg.matrix_rank(matrix):
                linear_Independent = True
                print('Vectors are linearly independent')
            else:
                print('Vectors are not linearly independent')
        except e:
            print(e)
        return linear_Independent


class BasisFinder(LinearSystemsBase):
    def _get_system_type(self):
        return 'basis_finder'

    def solve(self, **kwargs):
        self._a = kwargs.get('a')
        if self._a is None:
            raise ValueError('Input matrix a is not provided')
        self._pivoting = kwargs.get('pivoting') if kwargs.get('pivoting') is not None else True
        self._rounding_digit = kwargs.get('rounding_digit') if kwargs.get('rounding_digit') is not None else 10
        print(f'Input Matrix: \n{self._a}\n\n')
        print(f'Input Matrix Rank: \n{np.linalg.matrix_rank(self._a)}\n\n')
        self.reduce_to_ref()
        self.find_rs_basis()
        self.find_cs_basis()
        self.find_ns_basis()

    def find_rs_basis(self):
        self.rs_basis = self._ref_a[np.nonzero(np.sum(self._ref_a, axis =1))]
        print(f'\n\nBasis for Row space (Non zero rows in REF):\n{self.rs_basis}')

    def find_cs_basis(self):
        self.pivot_index_list = [np.argwhere(x != 0)[0][0] for x in self.rs_basis]
        cs_basis = self._a.T[self.pivot_index_list]
        print(f'\n\nBasis for Column space (Columns in input matrix corresponding to pivot entries in REF):\n{cs_basis}')

    def find_ns_basis(self):
        M = Matrix(self._a)
        #print("Matrix : {} ".format(M))

        # Use sympy.nullspace() method
        M_nullspace = M.nullspace()

        a_nullspace = np.array(M_nullspace)
        basis_list = [row.flatten() for row in a_nullspace]
        ns_basis = np.array(basis_list)
        var_index = range(self._a.shape[1])
        nullity = self._a.shape[1] - np.linalg.matrix_rank(self._a)
        print(f'\n\nNullity of the given Matrix is: {nullity}')
        if nullity > 0:
            free_vars = [f'x{x + 1}' for x in var_index if x not in self.pivot_index_list]
            print(f'This means there are {nullity} free variables: {free_vars} ')
            print(f'Setting {free_vars} with arbitrary values.')
        print(f'\nWe get basis for Null space (Solution of homogenous equation Ax = 0):\n{ns_basis}')

    def reduce_to_ref(self):
        print('Reducing the matrix to REF:\n')
        a = self._a.copy().astype(float)
        num_rows = a.shape[0]
        for i in range(num_rows):
            if self._pivoting == True:
                LinearSystemsHelper.pivot(a, i)

            if np.argwhere(a[i] != 0).size == 0:
                break

            pivot_index = np.argwhere(a[i] != 0)[0][0]
            if pivot_index == a.shape[1] -1:
                break

            for j in range(i + 1, num_rows):
                if a[j, pivot_index] != 0 and a[i, pivot_index] != 0:
                    print(f'R{j + 1} -> R{j + 1} - {a[j, pivot_index]}/{a[i, pivot_index]}*R{i + 1}')
                    a[j] -= LinearSystemsHelper.round_to_n(LinearSystemsHelper.round_to_n(a[j, pivot_index] / a[i, pivot_index],
                                                      self._rounding_digit) * LinearSystemsHelper.round_to_n(a[i],
                                                                                   self._rounding_digit), self._rounding_digit)
                    a[j] = LinearSystemsHelper.round_to_n(a[j], self._rounding_digit)
                    a[j, pivot_index] = 0
                    print(f'Output:\n{a}\n\n')

        print(f'Final REF form: \n {a}')
        self._ref_a = a


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

            if np.argwhere(a[i] != 0).size == 0:
                break

            pivot_index = np.argwhere(a[i] != 0)[0][0]
            if pivot_index == a.shape[1] - 1:
                break
            for j in range(i + 1, num_rows):
                if a[j, pivot_index] != 0 and a[i, pivot_index] != 0:
                    print(f'R{j + 1} -> R{j + 1} - {a[j, pivot_index]}/{a[i, pivot_index]}*R{i + 1}')
                    a[j] -= a[j, pivot_index] / a[i, pivot_index] *a[i]
                    a[j, pivot_index] = 0
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
        #if num_vars <= num_rows:
        # for i in range(num_rows):
        #     if sum(a[i, :-1]) <= 1.0e-12 and a[i, num_vars] != 0:
        #         print('Given system of equations is inconsistent')
        #         return
        if np.linalg.matrix_rank(a[:,:-1]) != np.linalg.matrix_rank(a):
            print('Given system of equations is inconsistent')
            return
        elif np.linalg.matrix_rank(a) < num_vars:
            print(f'There are {num_vars - np.linalg.matrix_rank(a)} free variables.\nGiven system of equations have infinitely many solutions')
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
        # elif num_vars > num_rows:
        #     print('Given system of equations have infinitely many solutions')


class GaussEliminationRounding(LinearSystemsBase):

    def _get_system_type(self):
        return 'gauss_elimination_rounding'

    def solve(self, **kwargs):
        print('Starting Gauss Elimination with rounding')
        self._a = kwargs.get('a')
        self._b = kwargs.get('b')
        self._pivoting = kwargs.get('pivoting')
        self._rounding_digit = kwargs.get('rounding_digit') if kwargs.get('rounding_digit') is not None else 4
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

            if np.argwhere(a[i] != 0).size == 0:
                break

            pivot_index = np.argwhere(a[i] != 0)[0][0]
            if pivot_index == a.shape[1] -1:
                break

            for j in range(i + 1, num_rows):
                if a[j, pivot_index] != 0 and a[i, pivot_index] != 0:
                    print(f'R{j + 1} -> R{j + 1} - {a[j, pivot_index]}/{a[i, pivot_index]}*R{i + 1}')
                    a[j] -= LinearSystemsHelper.round_to_n(LinearSystemsHelper.round_to_n(a[j, pivot_index] / a[i, pivot_index],
                                                      self._rounding_digit) * LinearSystemsHelper.round_to_n(a[i],
                                                                                   self._rounding_digit), self._rounding_digit)
                    a[j] = LinearSystemsHelper.round_to_n(a[j], self._rounding_digit)
                    a[j, pivot_index] = 0
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
        #if num_vars <= num_rows:
        # for i in range(num_rows):
        #     if sum(a[i, :-1]) <= 1.0e-12 and a[i, num_vars] != 0:
        #         print('Given system of equations is inconsistent')
        #         return
        if np.linalg.matrix_rank(a[:, :-1]) != np.linalg.matrix_rank(a):
            print('Given system of equations is inconsistent')
            return
        elif np.linalg.matrix_rank(a) < num_vars:
            print(f'There are {num_vars - np.linalg.matrix_rank(a)} free variables.\nGiven system of equations have infinitely many solutions')
            return

        var_names = []
        var_counter = num_vars
        var_values = np.ones(num_vars + 1)
        # print(var_values)
        for j in range(num_vars - 1, -1, -1):
            if len(var_names) == 0:
                row = a[j]
                row_op = LinearSystemsHelper.round_to_n(row * var_values, self._rounding_digit)
                rhs = LinearSystemsHelper.round_to_n(sum(row_op) - row_op[j] - row_op[-1], self._rounding_digit)
                var_val = (row_op[-1] - rhs) / row_op[j]
                var_val = LinearSystemsHelper.round_to_n(var_val, self._rounding_digit)
                print(f'From R{j + 1}, we get X{j + 1} = {var_val}')
                var_values[j] = var_val
                var_names.append(f'X{j + 1}')
            else:
                row = a[j]
                row_op = LinearSystemsHelper.round_to_n(row * var_values, self._rounding_digit)
                rhs = LinearSystemsHelper.round_to_n(sum(row_op) - row_op[j] - row_op[-1], self._rounding_digit)
                var_val = (row_op[-1] - rhs) / row_op[j]
                var_val = LinearSystemsHelper.round_to_n(var_val, self._rounding_digit)
                var_list = list(var_values)
                var_list.reverse()
                var_list = var_list[1:len(var_names) + 1]
                var_str = [x + ' = ' + str(y) for x, y in zip(var_names, var_list)]
                print(f'Substituting {var_str} in R{j + 1}, we get X{j + 1} = {var_val}')
                var_values[j] = var_val
                var_names.append(f'X{j + 1}')
        # elif num_vars > num_rows:
        #     print('Given system of equations have infinitely many solutions')


if __name__ == '__main__':
    # a = np.array([[3,2,1], [2,1,1], [6,2,4] ])
    # b =np.array([[3,0,6]])
    # a = np.array([[1,1,-1], [1,1,1]])
    # b = np.array([[2,3]])
    # a = np.array([[0.0004, 1.402], [0.4003, -1.502]])
    # b = np.array([[1.406, 2.501]])
    a = np.array([[0,2,0,1],[2,2,3,2],[4,-3,0,1], [6,1,-6,-5]])
    b= np.array([[0,-2,-7,6]])
    #a = np.array([[4.5, 3.55],[4.5,2.8]])
    #a = np.array([[2,-1,1],[1,0,1],[3,-1,4]])
    lsf = LinearSystemsFactory()
    #lsf.get_linear_system(system_type='gauss_elimination').solve(a=a, b = b, pivoting = True)

    #lsf.get_linear_system(system_type='gauss_elimination_rounding').solve(a=a, b=b, pivoting = True,rounding_digit = 3)

    #lsf.get_linear_system(system_type='condition_number').solve(a=a, b=b, pivoting=True, rounding_digit=4)

    # a = np.array([[3,4,0,2],[2,-1,3,7],[1,16,-12,-22]])
    # lsf.get_linear_system(system_type='linear_independence_checker').solve(v1=[0,1,1],v2=[1,1,1], v3=[0,0,1])

    # a = np.array([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
    # lsf.get_linear_system(system_type='basis_finder').solve(a=a, pivoting = False, rounding_digit=10)

    #a = np.array([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
    # lsf.get_linear_system(system_type='gauss_seidel').solve(size = (4,4), diag_dom = True, randomize=True, rounding_digit= 4,
    #                                                         print_iterations = True, n_iterations =100, tolerance = 0.000001)

    # A, b = LinearSystemsHelper.generate_random_matrix((4,4), True)
    #
    # lsf.get_linear_system(system_type='gauss_seidel').solve(A = A, b = b, size=(4, 4), diag_dom=True, randomize=True,
    #                                                         rounding_digit=4,
    #                                                         print_iterations=True, n_iterations=100, tolerance=0.000001)
    #
    # lsf.get_linear_system(system_type='gauss_jacobi').solve(A = A, b = b, size=(4, 4), diag_dom=True, randomize=True,
    #                                                         rounding_digit=4,
    #                                                         print_iterations=True, n_iterations=100, tolerance=0.000001)

    A = np.array([[11, 5, 1], [5, 8, 2], [1, 3, 5]])

    lsf.get_linear_system(system_type='rayleigh_quotient').solve(A=A, b=b, size=(4, 4), diag_dom=True, randomize=True,
                                                            rounding_digit=8,
                                                            print_iterations=True, n_iterations=100, tolerance=0.001)

Factory for Linear Algebra algos for MFDS:
Master List of Algos implemented:
1. Gauss Elimination with and without partial pivoting
2. LU decomposition -  Crout , Cholesky, Dolittle
3. Gauss Seidel and Gauss jacobi
4. Diagonalization, power method and Rayleigh quotient
5. QR Factorization
6. Gram schmidt orthonormalization
6. SVD
7. LPP - graph and simplex


Steps for integration with Factory:
1. Extend the class LinearSystemsBase and implement the contract.
2. Add the system_type from your class to VALID_SYSTEM_TYPES list.


Other Algos:
1. Steepest descent with constant and variable learning rates
2. Finding maxima and minima using Hessian Matrix
3. Constrained optimization using Langrange Multiplier

Factory for Linear Algebra algos for MFDS:
Master List of Algos to be integrated:
1. GE with and without partial pivoting
2. LU  Crout , Cholesky, Dolittle
3. GS and GJ
4. Diagonalization, power method and Rayleigh quotient
5. QR
6. SVD
7. LPP - graph and simplex


Steps for integration with Factory:
1. Extend the class LinearSystemsBase and implement the contract.
2. Add the system_type from your class to VALID_SYSTEM_TYPES list.

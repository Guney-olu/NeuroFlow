import sympy as sp

class HessianCalculator:
    def __init__(self, function, variables):
        self.function = function
        self.variables = [sp.symbols(var) for var in variables]
        self.expr = sp.sympify(function(*self.variables))
        self.hessian_matrix = None

    def calculate_hessian_matrix(self, point):
        num_variables = len(self.variables)
        hessian_matrix = [[0] * num_variables for _ in range(num_variables)]

        for i, row_var in enumerate(self.variables):
            for j, col_var in enumerate(self.variables):
                partial_derivative_i = sp.diff(self.expr, row_var)
                partial_derivative_j = sp.diff(partial_derivative_i, col_var)

                hessian_matrix[i][j] = partial_derivative_j.evalf(subs=dict(zip(self.variables, point)))

        self.hessian_matrix = hessian_matrix
        return hessian_matrix

    def print_hessian_matrix(self):
        if self.hessian_matrix is None:
            print("Hessian matrix not calculated yet. Call calculate_hessian_matrix() first.")
        else:
            print("Hessian Matrix:")
            for row in self.hessian_matrix:
                print(row)
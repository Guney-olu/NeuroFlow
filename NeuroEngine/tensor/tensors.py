import random

class Tensor:
    def __init__(self, data):
        self.data = data
        self._calculate_shape()

    def _calculate_shape(self):
        if isinstance(self.data[0], (list, tuple)):
            self.shape = (len(self.data), len(self.data[0]))
        else:
            self.shape = (len(self.data),)

    def __add__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shapes must be the same for addition.")
            
            if len(self.shape) == 1:
                result_data = [self.data[i] + other.data[i] for i in range(self.shape[0])]
            else:
                result_data = [
                    [self.data[i][j] + other.data[i][j] for j in range(self.shape[1])]
                    for i in range(self.shape[0])
                ]
            
            return Tensor(result_data)
        else:
            if len(self.shape) == 1:
                result_data = [self.data[i] + other for i in range(self.shape[0])]
            else:
                result_data = [[self.data[i][j] + other for j in range(self.shape[1])] for i in range(self.shape[0])]
            
            return Tensor(result_data)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Shapes must be the same for subtraction.")
            
            if len(self.shape) == 1:
                result_data = [self.data[i] - other.data[i] for i in range(self.shape[0])]
            else:
                result_data = [
                    [self.data[i][j] - other.data[i][j] for j in range(self.shape[1])]
                    for i in range(self.shape[0])
                ]
            
            return Tensor(result_data)
        else:
            if len(self.shape) == 1:
                result_data = [self.data[i] - other for i in range(self.shape[0])]
            else:
                result_data = [[self.data[i][j] - other for j in range(self.shape[1])] for i in range(self.shape[0])]
            
            return Tensor(result_data)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Invalid shapes for matrix multiplication.")
            
            if len(self.shape) == 1:
                result_data = [self.data[i] * other.data[i] for i in range(self.shape[0])]
            else:
                result_data = [
                    [self.data[i][k] * other.data[k][j] for k in range(self.shape[1])]
                    for j in range(other.shape[1])
                ]
            
            return Tensor(result_data)
        else:
            if len(self.shape) == 1:
                result_data = [self.data[i] * other for i in range(self.shape[0])]
            else:
                result_data = [[self.data[i][j] * other for j in range(self.shape[1])] for i in range(self.shape[0])]
            
            return Tensor(result_data)

    def __truediv__(self, scalar):
        if len(self.shape) == 1:
            result_data = [self.data[i] / scalar for i in range(self.shape[0])]
        else:
            result_data = [[self.data[i][j] / scalar for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        return Tensor(result_data)

    def __pow__(self, exponent):
        if len(self.shape) == 1:
            result_data = [self.data[i] ** exponent for i in range(self.shape[0])]
        else:
            result_data = [[self.data[i][j] ** exponent for j in range(self.shape[1])] for i in range(self.shape[0])]
        
        return Tensor(result_data)

    def __repr__(self):
        return f"<class 'neuroflow.Tensor'>({self.data})"

    @classmethod
    def empty(cls, *shape):
        data = [[0.0] * shape[-1] for _ in range(shape[-2])]
        return cls(data)

    @classmethod
    def zeros(cls, *shape):
        data = [[0.0] * shape[-1] for _ in range(shape[-2])]
        return cls(data)

    @classmethod
    def ones(cls, *shape):
        data = [[1.0] * shape[-1] for _ in range(shape[-2])]
        return cls(data)

    @classmethod
    def rand(cls, *shape):
        data = [[random.random() for _ in range(shape[-1])] for _ in range(shape[-2])]
        return cls(data)

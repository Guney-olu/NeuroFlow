# NeuroFlow

![Image Alt text](NeuroEngine/assests/ImgNf.jpeg)

NeuroFlow: Dive into the stream of deep learning with ease. Empowering your neural network creations to flow effortlessly from idea to implementation

### How to Use
from NeuroEngine.nn_architecture import NeuralNetwork
from NeuroEngine.visualize import visualize_neural_network
import numpy as np

```python
# Using Neuroflow tensors
from NeuroEngine.tensor.tensors import Tensor

tensor1 = Tensor([1, 2, 3])
tensor2 = Tensor([4, 5, 6])

result = tensor1 + tensor2

print(result.data)

x = Tensor.empty(3, 4)
print("Tensor with zeros:")
print(x.data)

zeros = Tensor.zeros(2, 3)
print("Tensor with zeros:")
print(zeros)

ones = Tensor.ones(2, 3)
print("Tensor with ones:")
print(ones)

random_tensor = Tensor.rand(2, 3)
print("Random Tensor:")
print(random_tensor)

# All operations are also supported 
ones = Tensor.zeros(2, 2) + 1
twos = Tensor.ones(2, 2) * 2
threes = (Tensor.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

```


```python
from NeuroEngine.nn_architecture import NeuralNetwork
from NeuroEngine.visualize import visualize_neural_network
import numpy as np

# Create synthetic data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create an instance of your neural network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Visualize the neural network
visualize_neural_network(nn)

# Train the neural network
nn.train(X, y, epochs=10000)

# Make predictions
predictions = nn.predict(X)
print(predictions)
```

### Goal

* Simple Implementation: A library designed for simplicity, making it easy for beginners to understand and implement neural networks.

* Visualization: visualize_neural_network function which allows us to visualize the architecture of our neural network, making it easier to understand its structure.

* Flexibility: Customizing the neural network architecture by specifying the input size, hidden layer size, and output size.

### Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or provide feedback.

### License

MIT
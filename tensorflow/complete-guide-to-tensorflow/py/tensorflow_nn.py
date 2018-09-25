#/usr/bin

import numpy as np

# Manual Neural Network

class SimpleClass():

    def __init__(self, str_input):
        print("SIMPLE"+str_input)


class ExtendedClass(SimpleClass):

    def __init__(self):
        super().__init__(" My String")
        print('Extended')


s = ExtendedClass()


####### Operation

class Operation():
    def __init__(self, input_nodes = []):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:

            node.output_nodes.append(self)

        _default_graph.operations.append(self)


    def compute(self):
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):
    def __init__(self, a, b):
        super().__init__([a, b])

    def compute(self, a_var, b_var):
        self.inputs = [a_var, b_var]
        return a_var * b_var


class matmul(Operation):
    def __init__(self, a, b):

        super().__init__([a, b])

    def compute(self, a_mat, b_mat):

        self.inputs = [a_mat, b_mat]
        return a_mat.dot(b_mat)


class Placeholder():
    """
    A placeholder is a node that needs to be provided a value for computing the output in the Graph.
    """

    def __init__(self):

        self.output_nodes = []

        _default_graph.placeholders.append(self)


class Variable():
    """
    This variable is a changeable parameter of the Graph.
    """

    def __init__(self, initial_value = None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


class Graph():
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


g = Graph()
g.set_as_default()

A = Variable(10)
b = Variable(1)
x = Placeholder()

y = multiply(A,x)
z = add(y,b)



def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:

    def run(self, operation, feed_dict = {}):
        """
          operation: The operation to compute
          feed_dict: Dictionary mapping placeholders to input values (the data)
        """

        # Puts nodes in correct order
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:

            if type(node) == Placeholder:

                node.output = feed_dict[node]

            elif type(node) == Variable:

                node.output = node.value

            else: # Operation

                node.inputs = [input_node.output for input_node in node.input_nodes]


                node.output = node.compute(*node.inputs)

            # Convert lists to numpy arrays
            if type(node.output) == list:
                node.output = np.array(node.output)

        # Return the requested node value
        return operation.output



sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
print(result)


################################################


g = Graph()

g.set_as_default()

A = Variable([[10,20],[30,40]])
b = Variable([1,2])

x = Placeholder()

y = matmul(A,x)

z = add(y,b)

sess = Session()
result = sess.run(operation=z,feed_dict={x:10})
print(result)


################## SIGMOID
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)

plt.plot(sample_z, sample_a)
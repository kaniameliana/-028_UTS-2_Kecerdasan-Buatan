# Kania Meliana Fityanti_21091397028
# Multi Neuron Batch Input dengan Input Layer Feature 10
# Per batch nya 6 Input
# Hidden Layer 1 yaitu 5 Neuron
# Hidden Layer 2 yaitu 3 Neuron

# inisialisasi numpy
import numpy as np

# inisialisasi variabel
# memasukan nilai variabel layer feature 10 dengan batch sejumlah 6 
inputs = [[0.5, 1.5, 2.5, 3, 4.5, 5, 1, 4, 3.5, 7],
		  [1.0, 2.0, 3.5, 3.0, 5.0, 6.0, 4.5, 1.5, 4.0, 2.5],
		  [2.5, 3.0, 4.5, 5.0, 1.0, 4.0, 3.5, 6.0, 9.0, 5.5],
		  [4.0, 4.5, 2.0, 1.5, 2.5, 9, 0.5, 7, 6, 3],
		  [6, 2.5, 9, 2, 1, 1.5, 7, 6.5, 3.5, 4],
		  [5, 3, 4.5, 2.5, 2, 9, 3.5, 4, 6, 1]]

# memberikan nilai bobot pada variabel sesuai dengan jumlah input
# memasukan jumlah weight sesuai dengan jumlah neuron yaitu sejumlah 5  
weights1 = [[0.5, 0.6, 0.47, -0.4, 0.31, 0.26, 0.4, 0.2, 0.15, -0.2],
		   [0.2, 0.51, 0.8, 0.3, 0.9, 0.7, 0.1, 0.6, 0.4, -0.3],
		   [0.15, 0.27, -0.2, 0.2, 0.16, -0.7, 0.31, 0.13, 0.18, 0.4],
		   [0.1, 0.3, 0.9, 0.21, 0.8, 0.35, 0.4, 0.37, 0.2, 0.7],
		   [0.3, -0.2, 0.46, 0.5, -0.4, 0.15, 0.25, 0.2, 0.13, -0.5]]

# inisialisasi biases pada layer1 sesuai dengan neuron yang ditentukan yaitu 5
biases1 =   [2, 3, 5, 0.5, 4]

# inisialisasi jumlah weight 2, weight layer 2 = neuron layer 1 yaitu 5
# memasukkan jumlah weight sesuai dengan neuron layer 2 yaitu 3 neuron
weights2 = [[-0.2, 1.5, 1.1, 2.1, -1.3],
			[0.4, 1.6, 2.4, 3.0, 2.2],
			[2.4, 2.5, 3.0, 1.1, 1.5]]

# inisialisasi biases pada layer2 dengan neuron yang ditentukan yaitu 3			
biases2 =  [1.0, 3.0, 0.5]

# output
# menghitung layer1 dengan (inputs*weight1) dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

# menghitung layer2 dengan hasil perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs) 
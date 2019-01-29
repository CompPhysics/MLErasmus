import numpy as np 
import tensorflow as tf

tf.set_random_seed(4155)

# Just to reset the graph such that it is possible to rerun this
# Jupyter cell without resetting the whole kernel.
tf.reset_default_graph()

Nx = 10
x = np.linspace(0,1, Nx)

x_tf = tf.convert_to_tensor(x.reshape(-1,1),dtype=tf.float64)

num_iter = 10000

num_hidden_neurons = [20,10]
num_hidden_layers = len(num_hidden_neurons)

# Input layer
previous_layer = x_tf
    
# Hidden layers
for l in range(num_hidden_layers):
    current_layer = tf.layers.dense(previous_layer, num_hidden_neurons[l], activation=tf.nn.sigmoid)
    previous_layer = current_layer
    
# Output layer
dnn_output = tf.layers.dense(previous_layer, 1)

g_trial = x_tf*(1 - x_tf)*dnn_output
d_g_trial = tf.gradients(g_trial,x_tf)
d2_g_trial = tf.gradients(d_g_trial,x_tf)

# f(x)
right_side = (3*x_tf + x_tf**2)*tf.exp(x_tf)

err = tf.square( -d2_g_trial[0] - right_side)
cost = tf.reduce_sum(err)

learning_rate = 0.001

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
traning_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    
    # Evaluate the initial cost:
    print('Initial cost: %g'%cost.eval())
    
    for i in range(num_iter):
        sess.run(traning_op)
    
    print('Final cost: %g'%cost.eval())
    
    g_dnn_tf_descent = g_trial.eval()



## Finite differences 
dx = 1/(Nx-1)

# Set up the matrix A
Nx2 = Nx - 2
A = np.zeros((Nx2,Nx2))

A[0,0] = 2
A[0,1] = -1

for i in range(1,Nx2-1):
    A[i,i-1] = -1
    A[i,i] = 2
    A[i,i+1] = -1

A[Nx2 - 1, Nx2 - 2] = -1
A[Nx2 - 1, Nx2 - 1] = 2

# Set up the vector f
def f(x):
    return (3*x + x**2)*np.exp(x)

f_vec = dx**2 * f(x[1:-1])

# Solve the equation
g_res = np.linalg.solve(A,f_vec)

# Insert the solution into an array
g_vec = np.zeros(Nx)
g_vec[1:-1] = g_res

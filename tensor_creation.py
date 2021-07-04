import tensorflow as tf

#tensor is a vector generalised to higher dimensions 
#examples
string = tf.Variable("This is my tensor",tf.string)
number = tf.Variable(324,tf.int32)
floating = tf.Variable(3.245,tf.float64)

print(string)
print(number)
print(floating)

# Each tensor has a Data type and a shape 
#Degree of a tensor / Rank of a tensor :- No. of the dimensions involved in the tensor 
rank_tensor = tf.Variable(["I am good"] , tf.string)
print(rank_tensor.shape)       # print the shape of the tensor
print(tf.rank(rank_tensor))    # print the rank of the tensor

# the shape of the tensor gives the how many dimensions are there and how many itens are there in each dimensions
#The tensor are always square.

tensor1 = tf.ones([1,2,3])   # defies a tensor having all elements 1 
# 1- interrrior list 
# 2- gives two list inside the bigger list
# 3- the number of the elements in each list

#print(tensor1)
print(tf.rank(tensor1))
print(tensor1.shape)

#changing the shape of the tensor 
tensor2 = tf.reshape(tensor1,[2,3,1])
tensor3 = tf.reshape(tensor2,[3 , -1]) # auto select the remaining value in this case it will give a tensor of shape [3,2]

print(tensor1)
print(tensor2)
print(tensor3)

#There are different types of the tensors like (1)Variable (2) Constant (3) Placeholder (4) Sparse Tensor 
#except the variable tensor all the other are immmutable i.e. their value does not change duting the excution.
#
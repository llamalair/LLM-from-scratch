import torch
import numpy as np

tensor0d = torch.tensor(1)

tensor1d = torch.tensor([1,2,3])

tensor2d = torch.tensor([[1,2,3],
                        [4,5,6]])

tensor3d = torch.tensor([[[1,2,3],[4,5,6]],
                        [[7,8,9],[10,11,12]]])


print("\n")
print(tensor2d)
print("\n")
print("To get the datatype of the tensor")
print(tensor2d.dtype)
print("\n")
print("Allow us to access the shape of the tensor")
print(tensor2d.shape)
print("\n")
print("To reshape it ur desire tensor shape: ")
print(tensor2d.reshape(3,2)) # will work regardless, copying the data if necessary to ensure the desired shape.
print(tensor2d.view(3,2)) # requires the original data to be contiguous and will fail if it isn’t
print("\n")
print("To transpose a tensor: ") # which means flipping it across its diagonal
print(tensor2d.T)
print("\n")
print("To multiply two tensor: ")
print(tensor2d.matmul(tensor2d.T))
print(tensor2d @ tensor2d.T)
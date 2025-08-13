# install the python bindings for tannic
# pip install pytannic

import socket 
from struct import pack, unpack
from torch import flatten
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets.mnist import MNIST
from pytannic.client import Client 
from pytannic.torch.serialization import serialize, deserialize

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
dataset = MNIST(download=True, root='data/mnist', train=False, transform=transform)

INDEX = 5 

if __name__ == "__main__":
    with Client('127.0.0.1', 8080) as client:
        request = dataset[INDEX][0].view(1,784)
        print("Expected: ", dataset[INDEX][1])
        client.send(serialize(request)) #sends a tensor over network to the tannic server.   
        response = client.receive()
        tensor = deserialize(response) #receivse an int tensor with the result. 
        print(tensor)
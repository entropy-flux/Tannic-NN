from dataclasses import dataclass
import struct
from torch import dtype
from torch import (
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    complex64,
    complex128,
)

def dcodeof(dtype: dtype):
    if dtype == int8:
        return 12
    elif dtype == int16:
        return 13
    elif dtype == int32:
        return 14
    elif dtype == int64:
        return 15
    elif dtype == float32:
        return 24
    elif dtype == float64:
        return 25
    elif dtype == complex64:
        return 37
    elif dtype == complex128:
        return 38
    else:
        raise ValueError("Unknown dtype")

def dtypeof(code: int) -> dtype:
    if code == 12:
        return int8
    elif code == 13:
        return int16
    elif code == 14:
        return int32
    elif code == 15:
        return int64
    elif code == 24:
        return float32
    elif code == 25:
        return float64
    elif code == 37:
        return complex64
    elif code == 38:
        return complex128
    else:
        raise ValueError("Unknown code")

@dataclass
class Header:
    FORMAT = "<I B B H Q"  # uint32, uint8, uint8, uint16, size_t
    magic: int
    protocol: int
    code: int
    checksum: int
    nbytes: int 

    def pack(self) -> bytes:
        """Pack the header fields into bytes (little-endian)."""
        return struct.pack(
            self.FORMAT,
            self.magic,
            self.protocol,
            self.code,
            self.checksum,
            self.nbytes, 
        )

    @classmethod
    def unpack(cls, data: bytes):
        """Unpack bytes into a Header instance (little-endian)."""
        unpacked = struct.unpack(cls.FORMAT, data)
        return cls(*unpacked)
    

 
@dataclass
class Metadata:
    FORMAT = "<B Q Q B" 
    dcode: int
    offset: int
    nbytes: int
    rank: int

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, self.dcode, self.offset, self.nbytes, self.rank)

    @classmethod
    def unpack(cls, data: bytes):
        return cls(*struct.unpack(cls.FORMAT, data))

    
import socket

class Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.socket = None

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def begin(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def close(self):
        if self.socket:
            self.socket.close()
            self.socket = None

    def send(self, data: bytes): 
        if not self.socket:
            raise RuntimeError("Socket not connected")
        self.socket.sendall(data)
 
     
    def receive(self) -> bytes: 
        hsize = struct.calcsize(Header.FORMAT)
        header_data = self._recvall(hsize)
        header = Header.unpack(header_data)
        if header.magic != MAGIC:
            raise ValueError("Invalid magic number in received data")
  
        payload = self._recvall(header.nbytes) 
        return header_data + payload
    
    def _recvall(self, size: int) -> bytes: 
        buf = b""
        while len(buf) < size:
            chunk = self.socket.recv(size - len(buf))
            if not chunk:
                raise ConnectionError("Socket closed before receiving enough data")
            buf += chunk
        return buf


MAGIC = (69 | (82 << 8) | (73 << 16) | (67 << 24)) 

from torch import Tensor 


def serialize(tensor: Tensor, code: int = 1) -> bytes:
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    rank = tensor.dim()
    shape  = tensor.shape
    buffer = tensor.numpy().tobytes()
    nbytes = struct.calcsize(Metadata.FORMAT) + rank * struct.calcsize("Q")  + len(buffer)
    header = Header(magic=MAGIC, protocol=1, code=code, checksum=0xABCD ,nbytes = nbytes)
    metadata = Metadata(dcode=dcodeof(tensor.dtype), offset=0, nbytes=len(buffer), rank=rank)  
    return header.pack() + metadata.pack() + struct.pack(f"<{rank}Q", *shape) + buffer 


def deserialize(data: bytes) -> Tensor:
    hsize = struct.calcsize(Header.FORMAT)
    msize = struct.calcsize(Metadata.FORMAT)
    
    metadata = Metadata.unpack(data[hsize:hsize + msize])
    shape = struct.unpack(f"<{metadata.rank}Q", data[hsize + msize: hsize + msize + metadata.rank * struct.calcsize("Q")])
    
    offset = hsize + msize + 8 * metadata.rank
    buffer = bytearray(data[offset: offset + metadata.nbytes])
    return torch.frombuffer(buffer, dtype=dtypeof(metadata.dcode)).reshape(shape)


from torcheval.metrics import Mean, MulticlassAccuracy
from torch.utils.data import DataLoader
from torch import Tensor
import torch
from torch import reshape
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets.mnist import MNIST

class Metrics:
    def __init__(self, device: str | None = None): 
        self.accuracy = MulticlassAccuracy(num_classes=10, device=device)
        
    def update(self, predictions: Tensor, targets: Tensor) -> None: 
        self.accuracy.update(predictions, targets)
        
    def compute(self) -> dict[str, Tensor]:
        return { 
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None: 
        self.accuracy.reset() 


if __name__ == "__main__":  
    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    
    mnist = MNIST(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset=mnist, batch_size=256)
    
    metrics = Metrics(device="cpu")
     
    with Client("127.0.0.1", 8080) as client:
        for batch_idx, (images, labels) in enumerate(loader): 
            data = serialize(images.view(images.size(0), -1))
            client.send(data) 
            predictions = deserialize(client.receive())
            metrics.update(predictions, labels)
            results = metrics.compute()
            print(f"Accuracy: {results['accuracy'].item():.4f}")

        print("Done!")
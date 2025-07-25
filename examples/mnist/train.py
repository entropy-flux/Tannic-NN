from typing import Iterable 
from torch import Tensor
from torch import argmax
from torch.nn import Module, Flatten 
from torch.nn import Linear, ReLU, Dropout
from torch.optim import Optimizer, Adam
from torch.nn import CrossEntropyLoss
from torch.nn import Dropout 
from torch.utils.data import DataLoader 
from torcheval.metrics import Mean, MulticlassAccuracy
from torchsystem import Aggregate 
from torchsystem import Depends 
from torchsystem.depends import Depends, Provider
from torchsystem.services import Service, Consumer, event
from torchsystem.compiler import Compiler  
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from mltracker import getallmodels
from mltracker.ports import Models 
 
class Metrics:
    def __init__(self, device: str | None = None):
        self.loss = Mean(device=device)
        self.accuracy = MulticlassAccuracy(num_classes=10, device=device) 
        
    def update(self, batch: int, loss: Tensor, predictions: Tensor, targets: Tensor) -> None:
        self.loss.update(loss)
        self.accuracy.update(predictions, targets)
        if batch % 200 == 0:
            print(f"--- Batch {batch}: loss={loss.item()}")
        
    def compute(self) -> dict[str, Tensor]:
        return {
            'loss': self.loss.compute(),
            'accuracy': self.accuracy.compute()
        }
    
    def reset(self) -> None:
        self.loss.reset()
        self.accuracy.reset() 

class Digits:
    def __init__(self, train: bool, normalize: bool):
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]) if normalize else ToTensor()
        self.data = MNIST(download=True, root='./data/mnist', train=train)
    
    def __getitem__(self, index: int):   
        return self.transform(self.data[index][0]), self.data[index][1]
    
    def __len__(self):
        return len(self.data)
    

class MLP(Module):
    def __init__(self, input_features: int, hidden_features: int, output_features: int, p: float = 0.5):
        super().__init__()
        self.epoch = 0
        self.input_layer = Linear(input_features, hidden_features)
        self.dropout = Dropout(p)
        self.activation = ReLU()
        self.output_layer = Linear(hidden_features, output_features)

    def forward(self, features: Tensor) -> Tensor:
        features = self.input_layer(features)
        features = self.dropout(features)
        features = self.activation(features)
        return self.output_layer(features) 

class Classifier(Aggregate):
    def __init__(self, hash: str, model: Module, criterion: Module, optimizer: Optimizer, metrics: Metrics):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.flatten = Flatten()
        self.hash = hash
        self.epoch = 0

    @property
    def id(self):
        return self.hash

    def forward(self, input: Tensor) -> Tensor:
        return self.model(self.flatten(input))
    
    def loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return self.criterion(outputs, targets)

    def fit(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return argmax(outputs, dim=1), loss
    
    def evaluate(self, inputs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]: 
        outputs = self(inputs)
        return argmax(outputs, dim=1), self.loss(outputs, targets)  
    

provider = Provider()
consumer = Consumer(provider=provider) 
service = Service(provider=provider)

def device() -> str:...

def models() -> Models:...

@service.handler
def train(model: Classifier, loader: Iterable[tuple[Tensor, Tensor]], device: str = Depends(device)):
    model.phase = 'train'
    for batch, (inputs, targets) in enumerate(loader, start=1): 
        inputs, targets = inputs.to(device), targets.to(device)  
        predictions, loss = model.fit(inputs, targets)
        model.metrics.update(batch, loss, predictions, targets)
    results = model.metrics.compute()
    consumer.consume(Trained(model, results))

@event
class Trained:
    model: Classifier 
    results: dict[str, Tensor]


@consumer.handler
def handle_epoch(event: Trained):
    event.model.epoch += 1

@consumer.handler
def handle_results(event: Trained, models: Models = Depends(models)):
    model = models.read(event.model.id)
    for name, metric in event.results.items():
        model.metrics.add(name, metric.item(), event.model.epoch, event.model.phase)

@consumer.handler
def print_metrics(event: Trained):
    print(f"-----------------------------------------------------------------")
    print(f"Epoch: {event.model.epoch}, Average loss: {event.results['loss'].item()}, Average accuracy: {event.results['accuracy'].item()}")
    print(f"-----------------------------------------------------------------")

@consumer.handler
def save_epoch(event: Trained, models: Models = Depends(models)):
    model = models.read(event.model.id) or models.create(event.model.id)
    model.epoch = event.model.epoch
 
compiler = Compiler[Classifier](provider=provider)

@compiler.step
def build_model(nn: Module, criterion: Module, optimizer: Module, metrics: Metrics, device: str = Depends(device)):
    print(f"Moving classifier to device {device}...")
    metrics.accuracy.to(device)
    metrics.loss.to(device)
    return Classifier('1', nn, criterion, optimizer, metrics).to(device)

@compiler.step
def compile_model(classifier: Classifier):
    print("Compiling model...")
    return compile(classifier)

@compiler.step
def bring_to_current_epoch(classifier: Classifier, models: Models = Depends(models)):
    print("Retrieving model from store...")
    model = models.read(classifier.id)
    if not model:
        print(f"model not found, creating one...")
        model = models.create(classifier.id, 'classifier')
    else:
        print(f"model found on epoch {model.epoch}")
    classifier.epoch = model.epoch
    return classifier 
 

if __name__ == '__main__': 
    repository = getallmodels('mlp')
    provider.override(device, lambda: 'cuda:0')
    provider.override(models, lambda: repository) 

    nn = MLP(784, 512, 10, 0.4)
    criterion = CrossEntropyLoss()
    optimizer = Adam(nn.parameters(), lr=0.001)
    metrics = Metrics()
    classifier = compiler.compile(nn, criterion, optimizer, metrics)
    datasets = {
        'train': Digits(train=True, normalize=True),
        'evaluation': Digits(train=False,  normalize=True),
    }
    loaders = {
        'train': DataLoader(datasets['train'], batch_size=64, shuffle=True, pin_memory=True, pin_memory_device='cuda', num_workers=4),
        'evaluation': DataLoader(datasets['evaluation'], batch_size=64, shuffle=False, pin_memory=True, pin_memory_device='cuda', num_workers=4)
    }

    for epoch in range(40):
        train(classifier, loaders['train'])
import numpy as np
from tinygrad.helpers import Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.dtype import dtypes 
from extra.datasets import fetch_mnist
from tinygrad.nn.optim import SGD


def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()


class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x):
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x
  
  
net = TinyNet()
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

X_train, Y_train, X_test, Y_test = fetch_mnist()

with Tensor.train():
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_train.shape[0], size=(64))
    batch = Tensor(X_train[samp], requires_grad=False)
    # get the corresponding labels
    labels = Tensor(Y_train[samp])

    # forward pass
    out = net(batch)

    # compute loss
    loss = sparse_categorical_crossentropy(out, labels)

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = out.argmax(axis=-1)
    acc = (pred == labels).mean()

    if step % 100 == 0:
      print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")


with Timing("Time: "):
  avg_acc = 0
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0, X_test.shape[0], size=(64))
    batch = Tensor(X_test[samp], requires_grad=False)
    # get the corresponding labels
    labels = Y_test[samp]

    # forward pass
    out = net(batch)

    # calculate accuracy
    pred = out.argmax(axis=-1).numpy()
    avg_acc += (pred == labels).mean()
  print(f"Test Accuracy: {avg_acc / 1000}")
import mindspore
from mindspore import nn
from dataset import datapipe
from model import Network
from mindspore.dataset import MnistDataset

model = Network()
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load, _ = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)

batchsize = 64

train_dataset = MnistDataset(r'E:\datasets\MNIST_Data\train')
test_dataset = MnistDataset(r'E:\datasets\MNIST_Data\test')
train_dataset = datapipe(train_dataset, batchsize)
test_dataset = datapipe(test_dataset, batchsize)

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.SGD(model.trainable_params(), 1e-2)

# 1. Define forward function
def forward_fn(data, label):
    logits = model(data)
    loss = loss_fn(logits, label)
    return loss, logits

# 2. Get gradient function
grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

# 3. Define function of one-step training
def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    optimizer(grads)
    return loss

def train(model, dataset):
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")

def eval(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"eval: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    epochs = 3
    print("------------Start Training------------")
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(model, train_dataset)
        eval(model, test_dataset, loss_fn)
    print("------------End Training------------")

    # Save checkpoint
    mindspore.save_checkpoint(model, "model.ckpt")
    print("Saved Model to model.ckpt")


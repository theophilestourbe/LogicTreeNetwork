import torch
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
import torchvision
from LogicTreeNet import LogicTreeNet
import mnist_dataset


batch_size = 512
learning_rate = 0.01
weight_decay = 0.
tau = 6.5
input_bits = 2
k = 16
valid_set_size = 0.2



train_set = mnist_dataset.MNIST('./data-mnist', train=True, download=True) #, remove_border=args.dataset == 'mnist20x20')
test_set = mnist_dataset.MNIST('./data-mnist', train=False) #, remove_border=args.dataset == 'mnist20x20')

train_set_size = math.ceil((1 - valid_set_size) * len(train_set))
valid_set_size = len(train_set) - train_set_size
train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
#device = 'cpu'
model = LogicTreeNet(1, k, 10, 3, 3, 1.0, tau, device)
loss_fn = nn.CrossEntropyLoss(reduction='sum')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# training loop

def eval(model, loader, loss_fn, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        total_loss = 0
        total = 0
        count = 0
        for x, y in loader:
            x = x.to(device).round()
            y = y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.cpu().item()
            total += x.shape[0]
            count += (out.argmax(-1) == y).sum().cpu().item()
        model.train(mode=orig_mode)
    return total_loss / total, count / total

def train(model, loader, loss_fn, optim):
    model.train()
    total_loss = 0
    total = 0
    count = 0
    for x, y in tqdm(loader):
        x, y = x.to(torch.float32).to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.detach().cpu().item()
        total += x.shape[0]
        count += (out.argmax(-1) == y).sum().cpu().item()
    return total_loss / total, count / total


best_valid_eval_loss = float('inf')

epoch = 100

for e in range(1, epoch + 1):
    print(f'Epoch {e}')
    train_loss, train_acc = train(model, train_loader, loss_fn, optim)
    print(f'Train Loss: {train_loss:.6f} Train Acc: {100 * train_acc:.2f}')
    valid_loss_train_mode, valid_acc_train_mode = eval(model, validation_loader, loss_fn, True)
    print('Validation Train mode:')
    print(f'Loss {valid_loss_train_mode:.6f} Acc: {100 * valid_acc_train_mode:.2f}')
    valid_loss_eval_mode, valid_acc_eval_mode = eval(model, validation_loader, loss_fn, False)
    print('Validation Eval mode:')
    print(f'Loss {valid_loss_eval_mode:.6f} Acc: {100 * valid_acc_eval_mode:.2f}')
    if best_valid_eval_loss > valid_loss_eval_mode:
        best_valid_eval_loss = valid_loss_eval_mode
        torch.save(model.state_dict(), 'best_model.pth')
        print('Saved best model')

print('Testing best model:')
model.load_state_dict(torch.load('best_model.pth', weights_only=True))

test_loss, test_acc = eval(model, test_loader, loss_fn, False)

print(f'Test Loss: {test_loss:.6f} Train Acc: {100 * test_acc:.2f}')
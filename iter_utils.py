import time
import copy
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from torch import cuda

def learning_scheduler(optimizer, epoch, lr=0.001, lr_decay_epoch=10):
    lr = lr * (0.5**(epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print('Learning rate is set to {}'.format(lr))
    for param in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def train(model, train_loader, criterion, optimizer, init_lr=0.001, decay_epoch, n_epoch=20, batch_size=200):
    since = time.time()
    best_model = model
    best_acc = 0.0
    loss_curve = []

    for epoch in range(n_epoch):
        print('Epoch {}/{}'.format(epoch+1, n_epoch))
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = learning_scheduler(optimizer, epoch, lr=init_lr, lr_decay_epoch=decay_epoch)
                model.train(True)
            else:
                model.train(False)  

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if cuda.is_available():
                model.cuda()



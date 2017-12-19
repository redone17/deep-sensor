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

def train(model, train_loader, criterion, optimizer, init_lr=0.001, decay_epoch=10, n_epoch=20):
    since = time.time()
    best_model = model
    best_accuracy = 0.0
    loss_curve = []

    for epoch in range(n_epoch):
        print('Epoch {}/{}'.format(epoch+1, n_epoch))
        
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer = learning_scheduler(optimizer, epoch, lr=init_lr, lr_decay_epoch=decay_epoch)
                model.train(True)
            else:
                model.train(False)  
            
            running_loss = 0.0
            running_corrects = 0 
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if use_cuda:
                    inputs, target = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0]
                _, predicted = torch.max(outputs.data, 1)
                running_corrects += torch.sum(predicted==targets.data)
                loss_curve.append(loss.data[0])

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = running_corrects / len(train_loader)
            print('{} loss: {:.4f}, accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

            if phase == 'val' and epoch_accuracy>best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(model)
        print(' ')
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best validation accuracy: {:.4f}'.format(best_accuracy))
    return best_model, loss_curve

def test(model, test_loader):
    corrects = 0
    model = model.cpu()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        corrects += torch.sum(predicted==targets.data)
    accuracy = corrects / len(test_loader)
    return accuracy


import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from functions import *
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.autograd import Variable

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# set path
data_path = "/mnt/cephfs/smartauto/users/guoli.wang/jiachen.xue/anti_spoofing/data/CASIA-CeFA/phase1"
save_model_path = "./ResNetCRNN_ckpt/"

train_file = '../anno/train.txt'
test_file = '../anno/test.txt'

train_loader_mode = 'rand'
test_loader_mode = 'rand'

num_seq = 31

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
res_size = 256        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 2             # number of target category
epochs = 20        # training epochs
batch_size = 32
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info


def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        # output has dim = (batch, number of classes)
        output = rnn_decoder(cnn_encoder(X))

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze(
        ).numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            # (y_pred != output) get the index of the max log-probability
            y_pred = output.max(1, keepdim=True)[1]

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(
        all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    tn, fp, fn, tp = confusion_matrix(
        all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy()).ravel()

    apcer = fp / (tn + fp)
    bpcer = fn / (fn + tp)
    acer = (apcer + bpcer) / 2

    # show information
    # print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))
    print('Test set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%, APCER: {:.2f}%, BPCER: {:.2f}%, ACER: {:.2f}%'.format(
        len(all_y), test_loss, 100 * test_score, 100 * apcer, 100 * bpcer, 100 * acer))

    # save Pytorch models of best record
    # torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
train_params = {'batch_size': batch_size, 'shuffle': True,
                'num_workers': 4, 'pin_memory': True} if use_cuda else {}
if test_loader_mode == 'rand':
    test_params = {'batch_size': batch_size, 'shuffle': False,
                   'num_workers': 4, 'pin_memory': True} if use_cuda else {}
else:
    test_params = {'batch_size': 1, 'shuffle': False,
                   'num_workers': 4, 'pin_memory': True} if use_cuda else {}


train_transform = transforms.Compose([
    transforms.Resize([res_size, res_size]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

test_transform = transforms.Compose([
    transforms.Resize([res_size, res_size]),
    transforms.CenterCrop([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_set, valid_set = Dataset_CRNN(data_path, train_file, num_seq, train_loader_mode, transform=train_transform), \
    Dataset_CRNN(data_path, test_file, num_seq,
                 test_loader_mode, transform=test_transform)

train_loader = data.DataLoader(train_set, **train_params)
valid_loader = data.DataLoader(valid_set, **test_params)

# Create model
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2,
                            drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
        list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
        list(cnn_encoder.module.fc3.parameters()) + \
        list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
        list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
        list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(
        log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation(
        [cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    # A = np.array(epoch_train_losses)
    # B = np.array(epoch_train_scores)
    # C = np.array(epoch_test_losses)
    # D = np.array(epoch_test_scores)
    # np.save('./CRNN_epoch_training_losses.npy', A)
    # np.save('./CRNN_epoch_training_scores.npy', B)
    # np.save('./CRNN_epoch_test_loss.npy', C)
    # np.save('./CRNN_epoch_test_score.npy', D)

# # plot
# fig = plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(np.arange(1, epochs + 1), A[:, -1])  # train loss (on epoch end)
# plt.plot(np.arange(1, epochs + 1), C)         #  test loss (on epoch end)
# plt.title("model loss")
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['train', 'test'], loc="upper left")
# # 2nd figure
# plt.subplot(122)
# plt.plot(np.arange(1, epochs + 1), B[:, -1])  # train accuracy (on epoch end)
# plt.plot(np.arange(1, epochs + 1), D)         #  test accuracy (on epoch end)
# plt.title("training scores")
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['train', 'test'], loc="upper left")
# title = "./fig_ResNetCRNN.png"
# plt.savefig(title, dpi=600)
# # plt.close(fig)
# plt.show()

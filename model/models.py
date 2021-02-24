# -*- coding: utf-8 -*-

import torch
from optim.pytorchtools import EarlyStopping
import torch.nn as nn


class Model_SemiTime(torch.nn.Module):

  def __init__(self, backbone, feature_size=64, nb_class=3):
    super(Model_SemiTime, self).__init__()
    self.backbone = backbone
    self.relation_head = torch.nn.Sequential(
                             torch.nn.Linear(feature_size*2, 256),
                             torch.nn.BatchNorm1d(256),
                             torch.nn.LeakyReLU(),
                             torch.nn.Linear(256, 1))
    self.sup_head = torch.nn.Sequential(
        torch.nn.Linear(feature_size, nb_class),
        # torch.nn.BatchNorm1d(256),
        # torch.nn.LeakyReLU(),
        # torch.nn.Linear(256, nb_class),
        # torch.nn.Softmax(),
    )

  def aggregate(self, features_P,features_F, K):
    relation_pairs_list = list()
    targets_list = list()
    size = int(features_P.shape[0] / K)
    shifts_counter=1
    for index_1 in range(0, size*K, size):
      for index_2 in range(index_1+size, size*K, size):
        # Using the 'cat' aggregation function by default
        pos1 = features_P[index_1:index_1 + size]
        pos2 = features_F[index_2:index_2+size]
        pos_pair = torch.cat([pos1,
                              pos2], 1)  # (batch_size, fz*2)

        # Shuffle without collisions by rolling the mini-batch (negatives)
        neg1 = torch.roll(features_F[index_2:index_2 + size],
                          shifts=shifts_counter, dims=0)
        neg_pair1 = torch.cat([pos1, neg1], 1) # (batch_size, fz*2)

        relation_pairs_list.append(pos_pair)
        relation_pairs_list.append(neg_pair1)

        targets_list.append(torch.ones(size, dtype=torch.float32).cuda())
        targets_list.append(torch.zeros(size, dtype=torch.float32).cuda())

        shifts_counter+=1
        if(shifts_counter>=size):
            shifts_counter=1 # avoid identity pairs
    relation_pairs = torch.cat(relation_pairs_list, 0).cuda()  # K(K-1) * (batch_size, fz*2)
    targets = torch.cat(targets_list, 0).cuda()
    return relation_pairs, targets


  def run_test(self, predict, labels):
      correct = 0
      pred = predict.data.max(1)[1]
      correct = pred.eq(labels.data).cpu().sum()
      return correct, len(labels.data)

  def train(self, tot_epochs, train_loader, train_loader_label, val_loader, test_loader, opt):
    patience = opt.patience
    early_stopping = EarlyStopping(patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    optimizer = torch.optim.Adam([
                  {'params': self.backbone.parameters()},
        {'params': self.relation_head.parameters()},
        {'params': self.sup_head.parameters()},
    ], lr=opt.learning_rate)
    c_criterion = nn.CrossEntropyLoss()
    BCE = torch.nn.BCEWithLogitsLoss()

    epoch_max = 0
    acc_max=0
    best_acc=0

    for epoch in range(tot_epochs):
      self.backbone.train()
      self.relation_head.train()
      self.sup_head.train()

      acc_epoch=0
      acc_epoch_cls=0
      loss_epoch=0
      loss_epoch_label=0

      for i, data_labeled in enumerate(train_loader_label):
          optimizer.zero_grad()

          # labeled sample
          (x, target)=data_labeled
          x = x.cuda()
          target = target.cuda()
          output = self.backbone(x)
          output = self.sup_head(output)
          loss_label = c_criterion(output, target)

          loss = loss_label
          loss.backward()
          optimizer.step()

          loss_epoch_label += loss_label.item()

          # estimate the accuracy
          prediction = output.argmax(-1)
          correct = prediction.eq(target.view_as(prediction)).sum()
          accuracy = (100.0 * correct / len(target))
          acc_epoch += accuracy.item()

      for i, (data_augmented, data_P, data_F, _) in enumerate(train_loader):
        K = len(data_augmented) # tot augmentations
        # x = torch.cat(data_augmented, 0).cuda()
        x_P = torch.cat(data_P, 0).cuda()
        x_F = torch.cat(data_F, 0).cuda()

        optimizer.zero_grad()
        # forward pass (backbone)
        features_P = self.backbone(x_P)
        features_F = self.backbone(x_F)
        # aggregation function
        relation_pairs, targets = self.aggregate(features_P, features_F, K)

        # forward pass (relation head)
        score = self.relation_head(relation_pairs).squeeze()
        # cross-entropy loss and backward
        loss = BCE(score, targets)
        loss.backward()
        optimizer.step()
        # estimate the accuracy
        predicted = torch.round(torch.sigmoid(score))
        correct = predicted.eq(targets.view_as(predicted)).sum()
        accuracy = (100.0 * correct / float(len(targets)))
        acc_epoch_cls += accuracy.item()
        loss_epoch += loss.item()

      acc_epoch_cls /= len(train_loader)
      loss_epoch /= len(train_loader)
      acc_epoch /= len(train_loader_label)
      loss_epoch_label /= len(train_loader_label)

      if acc_epoch_cls>acc_max:
          acc_max = acc_epoch_cls
          epoch_max = epoch


      acc_vals = list()
      acc_tests = list()
      self.backbone.eval()
      self.sup_head.eval()
      with torch.no_grad():
          for i, (x, target) in enumerate(val_loader):
              x = x.cuda()
              target = target.cuda()

              output = self.backbone(x).detach()
              output = self.sup_head(output)
              # estimate the accuracy
              prediction = output.argmax(-1)
              correct = prediction.eq(target.view_as(prediction)).sum()
              accuracy = (100.0 * correct / len(target))
              acc_vals.append(accuracy.item())

          val_acc = sum(acc_vals) / len(acc_vals)
          if val_acc >= best_acc:
              best_acc = val_acc
              best_epoch = epoch
              for i, (x, target) in enumerate(test_loader):
                  x = x.cuda()
                  target = target.cuda()

                  output = self.backbone(x).detach()
                  output = self.sup_head(output)
                  # estimate the accuracy
                  prediction = output.argmax(-1)
                  correct = prediction.eq(target.view_as(prediction)).sum()
                  accuracy = (100.0 * correct / len(target))
                  acc_tests.append(accuracy.item())

              test_acc = sum(acc_tests) / len(acc_tests)

      print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
          epoch, val_acc, test_acc, best_epoch))
      early_stopping(val_acc, self.backbone)
      if early_stopping.early_stop:
          print("Early stopping")
          break

      if (epoch+1)%opt.save_freq==0:
        print("[INFO] save backbone at epoch {}!".format(epoch))
        torch.save(self.backbone.state_dict(), '{}/backbone_{}.tar'.format(opt.ckpt_dir, epoch))

      print('Epoch [{}][{}][{}] loss= {:.5f}; Epoch ACC.= {:.2f}%, CLS.= {:.2f}%, '
            'Max ACC.= {:.1f}%, Max Epoch={}' \
            .format(epoch + 1, opt.model_name, opt.dataset_name,
                    loss_epoch, acc_epoch,acc_epoch_cls, acc_max, epoch_max))
    return test_acc, acc_epoch_cls, best_epoch


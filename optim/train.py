# -*- coding: utf-8 -*-


import torch
import utils.transforms as transforms
from dataloader.ucr2018 import UCR2018, MultiUCR2018_Intra
import torch.utils.data as data
from optim.pytorchtools import EarlyStopping
from model.model_backbone import SimConv4
import utils.transforms as transforms_ts
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def run_test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

def semisupervised_train(x_train, y_train, x_val, y_val, x_test, y_test, opt):
    # construct data loader
    # Those are the transformations used in the paper
    prob = 0.2  # Transform Probability
    cutout = transforms_ts.Cutout(sigma=0.1, p=prob)
    jitter = transforms_ts.Jitter(sigma=0.2, p=prob)  # CIFAR10
    scaling = transforms_ts.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms_ts.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms_ts.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms_ts.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    if '2C' in opt.class_type:
        cut_piece = transforms.CutPiece2C(sigma=opt.piece_size)
        temp_class=2
    elif '3C' in opt.class_type:
        cut_piece = transforms.CutPiece3C(sigma=opt.piece_size)
        temp_class=3
    elif '4C' in opt.class_type:
        cut_piece = transforms.CutPiece4C(sigma=opt.piece_size)
        temp_class=4
    elif '5C' in opt.class_type:
        cut_piece = transforms.CutPiece5C(sigma=opt.piece_size)
        temp_class = 5
    elif '6C' in opt.class_type:
        cut_piece = transforms.CutPiece6C(sigma=opt.piece_size)
        temp_class = 6
    elif '7C' in opt.class_type:
        cut_piece = transforms.CutPiece7C(sigma=opt.piece_size)
        temp_class = 7
    elif '8C' in opt.class_type:
        cut_piece = transforms.CutPiece8C(sigma=opt.piece_size)
        temp_class = 8

    tensor_transform = transforms.ToTensor()
    train_transform_peice = transforms.Compose(transforms_targets)

    train_transform = transforms_ts.Compose(transforms_targets + [transforms_ts.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    train_set_labeled = UCR2018(data=x_train, targets=y_train, transform=train_transform)
    train_set_unlabeled = MultiUCR2018_Intra(data=x_train, targets=y_train, K=opt.K,
                               transform=train_transform_peice, transform_cut=cut_piece,
                               totensor_transform=tensor_transform)

    val_set = UCR2018(data=x_val, targets=y_val, transform=transform_eval)
    test_set = UCR2018(data=x_test, targets=y_test, transform=transform_eval)

    train_dataset_size = len(train_set_labeled)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    trainloader_label = torch.utils.data.DataLoader(train_set_labeled,
                                                       batch_size=128,
                                                       sampler=train_sampler)
    trainloader_unlabel = torch.utils.data.DataLoader(train_set_unlabeled, batch_size=128)

    val_loader_lineval = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    # loading the saved backbone
    backbone = SimConv4().cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, opt.nb_class).cuda()

    # linear_layer = torch.nn.Sequential(
    #     torch.nn.Linear(opt.feature_size, 256),
    #     torch.nn.BatchNorm1d(256),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(256, nb_class),
    #     torch.nn.Softmax(),
    # ).cuda()

    cls_head = torch.nn.Sequential(
        torch.nn.Linear(opt.feature_size*2, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, temp_class),
        torch.nn.Softmax(),
    ).cuda()

    optimizer = torch.optim.Adam([{'params': backbone.parameters()},
                  {'params': linear_layer.parameters()},
                  {'params': cls_head.parameters()}], lr=opt.learning_rate)

    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    torch.save(backbone.state_dict(), '{}/backbone_init.tar'.format(opt.ckpt_dir))

    best_acc = 0
    best_epoch = 0
    acc_epoch_cls = 0

    print('Semi-supervised Train')
    for epoch in range(opt.epochs):
        backbone.train()
        linear_layer.train()
        cls_head.train()

        acc_epoch = 0
        acc_epoch_cls = 0
        loss_epoch_label = 0
        loss_epoch_unlabel = 0
        loss=0

        for i, data_labeled in enumerate(trainloader_label):
            optimizer.zero_grad()

            # labeled sample
            (x, target)=data_labeled
            x = x.cuda()
            target = target.cuda()
            output = backbone(x)
            output = linear_layer(output)
            loss_label = CE(output, target)
            loss_epoch_label += loss_label.item()

            loss_label.backward()
            optimizer.step()

            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_epoch += accuracy.item()

        for i, data_unlabeled in enumerate(trainloader_unlabel):
            optimizer.zero_grad()

            # unlabeled time piece
            (x_piece1, x_piece2, target_temp,_)=data_unlabeled
            x_piece1 = torch.cat(x_piece1, 0).cuda()
            x_piece2 = torch.cat(x_piece2, 0).cuda()
            target_temp = torch.cat(target_temp, 0).cuda()
            features_cut0 = backbone(x_piece1)
            features_cut1 = backbone(x_piece2)
            features_cls = torch.cat([features_cut0, features_cut1], 1)
            c_output = cls_head(features_cls)
            correct_cls, length_cls = run_test(c_output, target_temp)
            loss_unlabel = CE(c_output, target_temp)
            loss_unlabel.backward()
            optimizer.step()

            loss_epoch_unlabel += loss_unlabel.item()

            accuracy_cls = 100. * correct_cls / length_cls
            acc_epoch_cls += accuracy_cls.item()

        acc_epoch /= len(trainloader_label)
        acc_epoch_cls /= len(trainloader_unlabel)
        loss_epoch_label /= len(trainloader_label)
        loss_epoch_unlabel /= len(trainloader_unlabel)

        print('[Train-{}][{}] loss_label: {:.5f}; \tloss_unlabel: {:.5f}; \t Acc label: {:.2f}% \t Acc unlabel: {:.2f}%' \
              .format(epoch + 1, opt.model_name, loss_epoch_label, loss_epoch_unlabel, acc_epoch, acc_epoch_cls))

        acc_vals = list()
        acc_tests = list()
        backbone.eval()
        linear_layer.eval()
        with torch.no_grad():
            for i, (x, target) in enumerate(val_loader_lineval):
                x = x.cuda()
                target = target.cuda()

                output = backbone(x).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (x, target) in enumerate(test_loader_lineval):
                    x = x.cuda()
                    target = target.cuda()

                    output = backbone(x).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, backbone)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(backbone.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))

    return test_acc, best_epoch


def supervised_train(x_train, y_train, x_val, y_val, x_test, y_test, opt):
    # construct data loader
    # Those are the transformations used in the paper
    prob = 0.2  # Transform Probability
    cutout = transforms_ts.Cutout(sigma=0.1, p=prob)
    jitter = transforms_ts.Jitter(sigma=0.2, p=prob)  # CIFAR10
    scaling = transforms_ts.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms_ts.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms_ts.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms_ts.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms_ts.Compose(transforms_targets + [transforms_ts.ToTensor()])
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = UCR2018(data=x_train, targets=y_train, transform=train_transform)
    val_set_lineval = UCR2018(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = UCR2018(data=x_test, targets=y_test, transform=transform_lineval)

    train_dataset_size = len(train_set_lineval)
    partial_size = int(opt.label_ratio * train_dataset_size)
    train_ids = list(range(train_dataset_size))
    np.random.shuffle(train_ids)
    train_sampler = SubsetRandomSampler(train_ids[:partial_size])

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval,
                                                       batch_size=128,
                                                       sampler=train_sampler)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False)

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, opt.nb_class).cuda()
    # linear_layer = torch.nn.Sequential(
    #     torch.nn.Linear(opt.feature_size, 256),
    #     torch.nn.BatchNorm1d(256),
    #     torch.nn.LeakyReLU(),
    #     torch.nn.Linear(256, nb_class),
    #     torch.nn.Softmax(),
    # ).cuda()

    optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()},
                  {'params': linear_layer.parameters()}], lr=opt.learning_rate)

    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    torch.save(backbone_lineval.state_dict(), '{}/backbone_init.tar'.format(opt.ckpt_dir))

    best_acc = 0
    best_epoch = 0

    print('Semi-supervised Train')
    for epoch in range(opt.epochs):
        backbone_lineval.train()
        linear_layer.train()

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data)
            output = linear_layer(output)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_trains.append(accuracy.item())

        print('[Train-{}][{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, opt.model_name, opt.dataset_name, loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        backbone_lineval.eval()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, backbone_lineval)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(backbone_lineval.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))

    return test_acc, best_epoch


def supervised_train2(x_train, y_train, x_val, y_val, x_test, y_test, nb_class, opt):
    # construct data loader
    # Those are the transformations used in the paper
    prob = 0.2  # Transform Probability
    cutout = transforms_ts.Cutout(sigma=0.1, p=prob)
    jitter = transforms_ts.Jitter(sigma=0.2, p=prob)  # CIFAR10
    scaling = transforms_ts.Scaling(sigma=0.4, p=prob)
    magnitude_warp = transforms_ts.MagnitudeWrap(sigma=0.3, knot=4, p=prob)
    time_warp = transforms_ts.TimeWarp(sigma=0.2, knot=8, p=prob)
    window_slice = transforms_ts.WindowSlice(reduce_ratio=0.8, p=prob)
    window_warp = transforms_ts.WindowWarp(window_ratio=0.3, scales=(0.5, 2), p=prob)

    transforms_list = {'jitter': [jitter],
                       'cutout': [cutout],
                       'scaling': [scaling],
                       'magnitude_warp': [magnitude_warp],
                       'time_warp': [time_warp],
                       'window_slice': [window_slice],
                       'window_warp': [window_warp],
                       'G0': [jitter, magnitude_warp, window_slice],
                       'G1': [jitter, time_warp, window_slice],
                       'G2': [jitter, time_warp, window_slice, window_warp, cutout],
                       'none': []}

    transforms_targets = list()
    for name in opt.aug_type:
        for item in transforms_list[name]:
            transforms_targets.append(item)

    train_transform = transforms_ts.Compose(transforms_targets + [transforms_ts.ToTensor()])
    transform_lineval = transforms.Compose([transforms.ToTensor()])

    train_set_lineval = UCR2018(data=x_train, targets=y_train, transform=train_transform)
    val_set_lineval = UCR2018(data=x_val, targets=y_val, transform=transform_lineval)
    test_set_lineval = UCR2018(data=x_test, targets=y_test, transform=transform_lineval)

    train_loader_lineval = torch.utils.data.DataLoader(train_set_lineval, batch_size=128, shuffle=True)
    val_loader_lineval = torch.utils.data.DataLoader(val_set_lineval, batch_size=128, shuffle=False)
    test_loader_lineval = torch.utils.data.DataLoader(test_set_lineval, batch_size=128, shuffle=False)

    # loading the saved backbone
    backbone_lineval = SimConv4().cuda()  # defining a raw backbone model

    # 64 are the number of output features in the backbone, and 10 the number of classes
    linear_layer = torch.nn.Linear(opt.feature_size, nb_class).cuda()
    optimizer = torch.optim.Adam([{'params': backbone_lineval.parameters()},
                  {'params': linear_layer.parameters()}], lr=opt.learning_rate)

    CE = torch.nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(opt.patience, verbose=True,
                                   checkpoint_pth='{}/backbone_best.tar'.format(opt.ckpt_dir))

    torch.save(backbone_lineval.state_dict(), '{}/backbone_init.tar'.format(opt.ckpt_dir))

    best_acc = 0
    best_epoch = 0

    print('Supervised Train')
    for epoch in range(opt.epochs):
        backbone_lineval.train()
        linear_layer.train()

        acc_trains = list()
        for i, (data, target) in enumerate(train_loader_lineval):
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            output = backbone_lineval(data)
            output = linear_layer(output)
            loss = CE(output, target)
            loss.backward()
            optimizer.step()
            # estimate the accuracy
            prediction = output.argmax(-1)
            correct = prediction.eq(target.view_as(prediction)).sum()
            accuracy = (100.0 * correct / len(target))
            acc_trains.append(accuracy.item())

        print('[Train-{}][{}] loss: {:.5f}; \t Acc: {:.2f}%' \
              .format(epoch + 1, opt.model_name, loss.item(), sum(acc_trains) / len(acc_trains)))

        acc_vals = list()
        acc_tests = list()
        backbone_lineval.eval()
        linear_layer.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader_lineval):
                data = data.cuda()
                target = target.cuda()

                output = backbone_lineval(data).detach()
                output = linear_layer(output)
                # estimate the accuracy
                prediction = output.argmax(-1)
                correct = prediction.eq(target.view_as(prediction)).sum()
                accuracy = (100.0 * correct / len(target))
                acc_vals.append(accuracy.item())

            val_acc = sum(acc_vals) / len(acc_vals)
            if val_acc >= best_acc:
                best_acc = val_acc
                best_epoch = epoch
                for i, (data, target) in enumerate(test_loader_lineval):
                    data = data.cuda()
                    target = target.cuda()

                    output = backbone_lineval(data).detach()
                    output = linear_layer(output)
                    # estimate the accuracy
                    prediction = output.argmax(-1)
                    correct = prediction.eq(target.view_as(prediction)).sum()
                    accuracy = (100.0 * correct / len(target))
                    acc_tests.append(accuracy.item())

                test_acc = sum(acc_tests) / len(acc_tests)

        print('[Test-{}] Val ACC:{:.2f}%, Best Test ACC.: {:.2f}% in Epoch {}'.format(
            epoch, val_acc, test_acc, best_epoch))
        early_stopping(val_acc, backbone_lineval)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(backbone_lineval.state_dict(), '{}/backbone_last.tar'.format(opt.ckpt_dir))

    return test_acc, best_epoch





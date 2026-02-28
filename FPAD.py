import pickle
import os
import torch
import numpy as np
import torch.optim as optim
import scipy.io as scio
import torch.nn.functional as F
import utils.model as models
from function_adv import hash_adv_img, hash_adv_tag
from utils.tools import save_model, image_normalization, image_restoration, CalcMap, run_fuzzy_cmeans, GenerateCode
from utils.loss import loss_function, loss_cons_function


def one_step(opt, Dcfg, Tr_I, Tr_T, Tr_L):
    bit = opt.bit
    data_set = opt.dataset
    tau = opt.tau

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    batch_size = 256
    epochs = 100
    learning_ratei = 0.0001
    learning_ratet = 0.0001
    weight_decay = 10 ** -5
    eps = 1e-5

    text_len = Dcfg.tag_dim
    nclass = Dcfg.num_label
    num_train = Tr_I.shape[0]

    vgg_path = '/home/workspace/Y/Dataset/imagenet-vgg-f.mat'
    pretrain_model = scio.loadmat(vgg_path)
    image_model = models.Image_Net(bit, pretrain_model)
    image_model.cuda()
    optimizer_image = torch.optim.SGD([{'params': image_model.cnn_f.parameters(), 'lr': learning_ratei},
                                       {'params': image_model.image_module.parameters(), 'lr': learning_ratei},
                                       {'params': image_model.hash_module.parameters(), 'lr': learning_ratei}],
                                      lr=learning_ratei, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_image, step_size=100, gamma=0.3, last_epoch=-1)

    text_model = models.Txt_Net(text_len, bit)
    text_model.cuda()
    optimizer_text = torch.optim.SGD([
        {'params': text_model.text_module.parameters(), 'lr': learning_ratet},
        {'params': text_model.hash_module.parameters(), 'lr': learning_ratet}],
        lr=learning_ratet, weight_decay=weight_decay)
    scheduler_text = torch.optim.lr_scheduler.StepLR(optimizer_text, step_size=100, gamma=0.3, last_epoch=-1)

    class_model = models.LabelNet(bit)
    class_model.cuda()
    optimizer_label = optim.SGD(class_model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler_l = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=100, gamma=0.1, last_epoch=-1)

    hidden = 1024
    loss_now = 100
    for epoch in range(epochs):
        class_model.train()
        image_model.train()
        text_model.train()
        scheduler_l.step()
        scheduler_text.step()
        scheduler.step()
        epoch_loss = 0.0
        index = np.random.permutation(num_train)

        image_feat = np.zeros((num_train, hidden))
        text_feat = np.zeros((num_train, hidden))
        fused_feat = np.zeros((num_train, hidden * 2))

        for i in range(num_train // batch_size + 1):
            end_index = min((i + 1) * batch_size, num_train)
            ind = index[i * batch_size: end_index]
            train_img = Tr_I[ind].type(torch.float).cuda()
            train_text = Tr_T[ind].type(torch.float).cuda()

            _, image_feat_batch = image_model(train_img)
            _, text_feat_batch = text_model(train_text)
            image_feat[ind] = image_feat_batch.detach().cpu()
            text_feat[ind] = text_feat_batch.detach().cpu()
            fused_feat[ind] = F.normalize((torch.cat((image_feat_batch, text_feat_batch), dim=1)).detach().cpu())

        centroids_img = np.zeros((nclass, hidden))
        centroids_txt = np.zeros((nclass, hidden))
        centroids_fused = np.zeros((nclass, hidden * 2))

        for i in range(nclass):
            indices = np.where(Tr_L[:, i] == 1)[0]
            if len(indices) > 0:
                samples_img = image_feat[indices]
                samples_txt = text_feat[indices]
                samples_fused = fused_feat[indices]
                centroids_img[i] = np.mean(samples_img, axis=0)
                centroids_txt[i] = np.mean(samples_txt, axis=0)
                centroids_fused[i] = np.mean(samples_fused, axis=0)

        cluster_img, U_img = run_fuzzy_cmeans(image_feat, centroids_img, nclass)
        cluster_txt, U_txt = run_fuzzy_cmeans(text_feat, centroids_txt, nclass)
        cluster_fused, U_fused = run_fuzzy_cmeans(fused_feat, centroids_fused, nclass)

        cluster_img = torch.tensor(cluster_img).type(torch.float).cuda()
        cluster_txt = torch.tensor(cluster_txt).type(torch.float).cuda()
        cluster_fused = torch.tensor(cluster_fused).type(torch.float).cuda()

        for i in range(num_train // batch_size + 1):
            end_index = min((i + 1) * batch_size, num_train)
            ind = index[i * batch_size: end_index]
            train_img = Tr_I[ind].type(torch.float).cuda()
            train_text = Tr_T[ind].type(torch.float).cuda()
            train_label = Tr_L[ind].type(torch.float).cuda()

            the_batch = len(ind)
            img_out, image_feat = image_model(train_img)
            text_out, text_feat = text_model(train_text)
            _, fused_feat = class_model(F.normalize((torch.cat((image_feat, text_feat), dim=1))))

            cluster_fused_1 = F.normalize((torch.cat((cluster_img, cluster_txt), dim=1)))

            cluster_fused_1_proto, _ = class_model(cluster_fused_1)
            cluster_fused_proto, class_feat = class_model(cluster_fused)
            u = 0.2
            cluster_proto = cluster_fused_1_proto * u + (1 - u) * cluster_fused_proto
            class_prototype = torch.tanh(cluster_proto)

            class_prototype_save = class_prototype.detach()

            logit_img_feat = F.cosine_similarity(F.normalize(image_feat, dim=1).unsqueeze(1),
                                                 F.normalize(cluster_txt, dim=1), dim=-1)
            our_logit_feat = (torch.exp(logit_img_feat * tau) * train_label)
            mu_logit_feat = (torch.exp(logit_img_feat * (1 - train_label) * tau).sum(1).view(-1, 1).expand(the_batch,
                                                                                                           train_label.size()[
                                                                                                               1]) + our_logit_feat)
            loss_img2ptxt_feat = - (
                    (torch.log(our_logit_feat / (mu_logit_feat + eps) + eps + 1 - train_label)).sum(1) / (
                    train_label + eps).sum(
                1)).sum()

            logit_text_feat = F.cosine_similarity(F.normalize(text_feat, dim=1).unsqueeze(1),
                                                  F.normalize(cluster_img, dim=1), dim=-1)
            our_logit_text_feat = torch.exp(logit_text_feat * tau) * train_label
            mu_logit_text_feat = torch.exp(logit_text_feat * (1 - train_label) * tau).sum(1).view(-1, 1).expand(
                the_batch,
                train_label.size()[
                    1]) + our_logit_text_feat
            loss_txt2pimg_feat = - (
                    (torch.log(our_logit_text_feat / (mu_logit_text_feat + eps) + eps + 1 - train_label)).sum(
                        1) / (train_label + eps).sum(1)).sum()

            logit_feat = F.cosine_similarity(F.normalize(fused_feat, dim=1).unsqueeze(1),
                                             F.normalize(class_feat, dim=1), dim=-1)
            our_logit_feat = (torch.exp(logit_feat * tau) * train_label)
            mu_logit_feat = (torch.exp(logit_feat * (1 - train_label) * tau).sum(1).view(-1, 1).expand(the_batch,
                                                                                                       train_label.size()[
                                                                                                           1]) + our_logit_feat)

            loss_fuse_feat = - ((torch.log(our_logit_feat / (mu_logit_feat + eps) + eps + 1 - train_label)).sum(1) / (
                    train_label + eps).sum(
                1)).sum()

            loss_feat = (loss_img2ptxt_feat + loss_txt2pimg_feat) + 0.5 * loss_fuse_feat

            logit = F.cosine_similarity(img_out.unsqueeze(1), class_prototype, dim=-1)
            our_logit = (torch.exp(logit * tau) * train_label)
            mu_logit = (torch.exp(logit * (1 - train_label) * tau).sum(1).view(-1, 1).expand(the_batch,
                                                                                             train_label.size()[
                                                                                                 1]) + our_logit)
            loss_img2class = - ((torch.log(our_logit / (mu_logit + eps) + eps + 1 - train_label)).sum(1) / (
                    train_label + eps).sum(
                1)).sum()

            logit_text = F.cosine_similarity(text_out.unsqueeze(1), class_prototype, dim=-1)
            our_logit_text = torch.exp(logit_text * tau) * train_label
            mu_logit_text = torch.exp(logit_text * (1 - train_label) * tau).sum(1).view(-1, 1).expand(the_batch,
                                                                                                      train_label.size()[
                                                                                                          1]) + our_logit_text
            loss_txt2class = - ((torch.log(our_logit_text / (mu_logit_text + eps) + eps + 1 - train_label)).sum(
                1) / (train_label + eps).sum(1)).sum()

            loss_hamm = loss_img2class + loss_txt2class

            loss_all = loss_hamm + loss_feat

            optimizer_image.zero_grad()
            optimizer_label.zero_grad()
            optimizer_text.zero_grad()
            loss_all.backward()
            optimizer_text.step()
            optimizer_label.step()
            optimizer_image.step()
            epoch_loss += loss_all.item() / the_batch
        print('[epoch: %3d/%3d][loss: %3.5f]' %
              (epoch + 1, epochs, epoch_loss / (num_train // batch_size + 1)))
        if epoch_loss / (num_train // batch_size + 1) <= loss_now:
            class_model.save("step1/class_model.pth")
            print("save code...")
            loss_now = epoch_loss / (num_train // batch_size + 1)
            U_tar = (U_img + U_txt + U_fused) / 3
            with open('./proto_code/' + data_set + '_' + str(bit) + '.pkl', 'wb') as f:
                pickle.dump(
                    {'class_code': class_prototype_save,
                     'degrees': U_tar}, f)


def GetProtoCode(Tr_I, Tr_T, Tr_L, image_model, text_model, text_len, batch_size, num_train, nclass, bit):
    class_model = models.LabelNet(bit)
    class_model.load("./checkpoints/step1/class_model.pth")
    class_model.cuda()
    hidden = 1024

    image_feat = np.zeros((num_train, hidden))
    text_feat = np.zeros((num_train, hidden))
    fused_feat = np.zeros((num_train, hidden * 2))

    for i in range(num_train // batch_size + 1):
        end_index = min((i + 1) * batch_size, num_train)
        index = np.random.permutation(num_train)
        ind = index[i * batch_size: end_index]
        train_img = Tr_I[ind].type(torch.float).cuda()
        train_text = Tr_T[ind].type(torch.float).cuda()

        _, image_feat_batch = image_model(train_img)
        _, text_feat_batch = text_model(train_text)
        image_feat[ind] = image_feat_batch.detach().cpu()
        text_feat[ind] = text_feat_batch.detach().cpu()
        fused_feat[ind] = F.normalize((torch.cat((image_feat_batch, text_feat_batch), dim=1)).detach().cpu())

        centroids_img = np.zeros((nclass, hidden))
        centroids_txt = np.zeros((nclass, hidden))
        centroids_fused = np.zeros((nclass, hidden * 2))

    for i in range(nclass):
        indices = np.where(Tr_L[:, i] == 1)[0]
        if len(indices) > 0:
            samples_img = image_feat[indices]
            samples_txt = text_feat[indices]
            samples_fused = fused_feat[indices]
            centroids_img[i] = np.mean(samples_img, axis=0)
            centroids_txt[i] = np.mean(samples_txt, axis=0)
            centroids_fused[i] = np.mean(samples_fused, axis=0)

    cluster_img, U_img = run_fuzzy_cmeans(image_feat, centroids_img, nclass)
    cluster_txt, U_txt = run_fuzzy_cmeans(text_feat, centroids_txt, nclass)
    cluster_fused, U_fused = run_fuzzy_cmeans(fused_feat, centroids_fused, nclass)

    U_mix = (U_img + U_txt + U_fused) / 3

    cluster_img = torch.tensor(cluster_img).type(torch.float).cuda()
    cluster_txt = torch.tensor(cluster_txt).type(torch.float).cuda()
    cluster_fused = torch.tensor(cluster_fused).type(torch.float).cuda()

    cluster_img = torch.tensor(cluster_img).type(torch.float).cuda()
    cluster_txt = torch.tensor(cluster_txt).type(torch.float).cuda()
    cluster_fused = torch.tensor(cluster_fused).type(torch.float).cuda()

    cluster_fused_1 = F.normalize((torch.cat((cluster_img, cluster_txt), dim=1)))
    cluster_fused_1_proto, _ = class_model(cluster_fused_1)
    cluster_fused_proto, _ = class_model(cluster_fused)
    u = 0.2
    cluster_proto = cluster_fused_1_proto * u + (1 - u) * cluster_fused_proto
    class_prototype = torch.tanh(cluster_proto)

    class_code = class_prototype.detach()

    return class_code, U_mix


def two_step(opt, Dcfg, Tr_I, Tr_T, Tr_L, Db_I, Db_T, Db_L, Te_I, Te_T, Te_L):
    data_set = opt.dataset
    bit = opt.bit
    tau = opt.tau
    text_len = Dcfg.tag_dim
    nclass = Dcfg.num_label

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    batch_size = 256
    epochs = 120
    weight_decay = 5e-4
    ImageModel_path = 'checkpoints/'
    TextModel_path = 'checkpoints/'
    learning_ratei = 0.005
    learning_ratet = 0.005

    num_database, num_train, num_test = Db_I.shape[0], Tr_I.shape[0], Te_I.shape[0]
    fp = './proto_code/' + data_set + '_' + str(bit) + '.pkl'
    with open(fp, 'rb') as f:
        all_train = pickle.load(f)
        label_code = torch.sign(all_train['class_code'])
        U_tar = (all_train['degrees'])

    vgg_path = '/home/workspace/Y/Dataset/imagenet-vgg-f.mat'
    pretrain_model = scio.loadmat(vgg_path)
    image_model = models.Image_Net(bit, pretrain_model)
    text_model = models.Txt_Net(text_len, bit)
    image_model.cuda()
    text_model.cuda()

    optimizer_img = optim.SGD(image_model.parameters(), lr=learning_ratei, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_img, step_size=300, gamma=0.3, last_epoch=-1)
    optimizer_text = optim.SGD(text_model.parameters(), lr=learning_ratet, weight_decay=weight_decay)
    scheduler_text = torch.optim.lr_scheduler.StepLR(optimizer_text, step_size=300, gamma=0.3, last_epoch=-1)
    adv_image = torch.zeros_like(Tr_I.type(torch.float))
    adv_txt = torch.zeros_like(Tr_T.type(torch.float))

    for epoch in range(epochs):
        image_model.train()
        text_model.eval()
        scheduler.step()
        epoch_loss = 0.0
        index = np.random.permutation(num_train)
        if epoch % 5 == 0 and epoch >= 10:
            for i in range(num_train // batch_size + 1):
                end_index = min((i + 1) * batch_size, num_train)
                ind = index[i * batch_size: end_index]
                train_img = Tr_I[ind].type(torch.float).cuda()
                train_label = Tr_L[ind].type(torch.float).cuda()

                image_norm = image_normalization(train_img)
                x_adv = hash_adv_img(image_model, image_norm,
                                     label_code, train_label,
                                     tau, epoch, adv_image[ind].cuda()
                                     )
                x_adv = x_adv.detach().cpu()
                x_adv_pixel = image_restoration(x_adv)
                adv_image[ind] = x_adv_pixel

            for i in range(num_train // batch_size):
                end_index = min((i + 1) * batch_size, num_train)
                ind = index[i * batch_size: end_index]
                train_text = Tr_T[ind].type(torch.float).cuda()
                train_label = Tr_L[ind].type(torch.float).cuda()

                y_adv = hash_adv_tag(text_model, train_text,
                                     label_code, train_label,
                                     tau, epoch, adv_txt[ind].cuda()
                                     )
                y_adv = y_adv.detach().cpu()
                adv_txt[ind] = y_adv
            Mix_I = torch.cat([Tr_I, adv_image])
            Mix_T = torch.cat([Tr_T, adv_txt])
            Mix_L = torch.cat([Tr_L, Tr_L])
            Mix_B, U_mix = GetProtoCode(Mix_I, Mix_T, Mix_L, image_model, text_model, text_len, batch_size,
                                        num_train * 2,
                                        nclass, bit)
            Mix_B = Mix_B.cuda()
            diff = np.abs(np.concatenate([U_tar, U_tar], axis=1) - U_mix)
            per_class_diff = diff.mean(axis=1)
            omega = per_class_diff / (per_class_diff.sum() + 1e-5)

            del Mix_I, Mix_T, Mix_L, U_mix

        for i in range(num_train // batch_size + 1):
            end_index = min((i + 1) * batch_size, num_train)
            ind = index[i * batch_size: end_index]
            train_img = Tr_I[ind].type(torch.float).cuda()
            train_label = Tr_L[ind].type(torch.float).cuda()

            x_adv_img = adv_image[ind].cuda()
            img_out, _ = image_model(train_img)
            img_noisy_out, _ = image_model(x_adv_img)
            loss_ben = loss_function(img_out, label_code, train_label, 1 - train_label, tau)

            loss_hash_adv = loss_function(img_noisy_out, label_code, train_label, 1 - train_label, tau)

            if epoch > 10:
                loss_cons = loss_cons_function(Mix_B, label_code, nclass, omega, tau)
                loss_all = loss_ben + opt.alpha * loss_hash_adv + loss_cons
            else:
                loss_all = loss_ben

            optimizer_img.zero_grad()
            loss_all.backward()
            optimizer_img.step()
            epoch_loss += loss_all.item()
        print(
            '[epoch: %3d/%3d][loss_image: %3.5f]' %
            (epoch + 1, epochs, epoch_loss / (num_train // batch_size + 1)))

        image_model.eval()
        text_model.train()
        scheduler_text.step()
        epoch_loss = 0.0
        for i in range(num_train // batch_size):
            end_index = min((i + 1) * batch_size, num_train)
            ind = index[i * batch_size: end_index]
            train_text = Tr_T[ind].type(torch.float).cuda()
            train_label = Tr_L[ind].type(torch.float).cuda()

            y_adv = adv_txt[ind].cuda()
            txt_out, _ = text_model(train_text)
            txt_noisy_out, _ = text_model(y_adv)

            loss_ben = loss_function(txt_out, label_code, train_label, 1 - train_label, tau)

            loss_hash_adv = loss_function(txt_noisy_out, label_code, train_label, 1 - train_label, tau)

            if epoch > 10:
                loss_cons = loss_cons_function(Mix_B, label_code, nclass, omega, tau)
                loss_all = loss_ben + opt.alpha * loss_hash_adv + loss_cons
            else:
                loss_all = loss_ben
            optimizer_text.zero_grad()
            loss_all.backward()
            optimizer_text.step()
            epoch_loss += loss_all.item()
        print(
            '[epoch: %3d/%3d][loss_text: %3.5f]' %
            (epoch + 1, epochs, epoch_loss / (num_train // batch_size)))
        if (epoch + 1) % 20 == 0:
            image_model.eval()
            text_model.eval()

            qi, qt = GenerateCode(image_model, text_model, Te_I, Te_T, num_test, bit)
            ri, rt = GenerateCode(image_model, text_model, Db_I, Db_T, num_database, bit)

            map_it = CalcMap(qi, rt, Te_L, Db_L, 50)
            map_ii = CalcMap(qi, ri, Te_L, Db_L, 50)
            map_ti = CalcMap(qt, ri, Te_L, Db_L, 50)
            map_tt = CalcMap(qt, rt, Te_L, Db_L, 50)
            print('Epoch:%3d map_i2t:%3.5f map_t2i:%3.5f', epoch + 1, map_it, map_ti)
            print('Epoch:%3d map_i2i:%3.5f map_t2t:%3.5f', epoch + 1, map_ii, map_tt)

            save_model(image_model, ImageModel_path + 'image_model_FPAD.pth')
            save_model(text_model, TextModel_path + 'text_model_FPAD.pth')
            image_model.train()
            text_model.train()

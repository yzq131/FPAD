import math
from utils.tools import image_restoration
from utils.loss import loss_function, probabiliby_mean_function
import torch


def hash_adv_img(model, query,
                 label_code, train_label,
                 tau, epoch, adv,
                 epsilon=8 / 255, step=3, iteration=20, randomize=True):
    delta = torch.zeros_like(query).cuda()

    noisy_output, _ = model(adv)
    per1, flag = probabiliby_mean_function(noisy_output, label_code, train_label, 1 - train_label)

    p = min((1 + 4 * per1) / 2, 1)

    if flag <= 0 and epoch > 10:
        delta = torch.zeros_like(query).cuda()
        iteration = math.ceil(iteration * p)
        epsilon = math.ceil(epsilon * p * 255) / 255
        if randomize:
            delta.uniform_(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
        delta.requires_grad = True
        for i in range(iteration):
            noisy_output, _ = model(image_restoration(query + delta))
            loss = loss_function(noisy_output, label_code, 1 - train_label, train_label, tau)
            loss.backward()

            delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
            delta.grad.zero_()
    else:
        if randomize:
            delta.uniform_(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
        delta.requires_grad = True
        for i in range(iteration):
            noisy_output, _ = model(image_restoration(query + delta))
            loss = loss_function(noisy_output, label_code, 1 - train_label, train_label, tau)
            loss.backward()

            delta.data = delta - step / 255 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(-1, 1) - query.data
            delta.grad.zero_()

    return query + delta.detach()


def hash_adv_tag(model, query,
                 label_code, train_label,
                 tau, epoch, adv,
                 epsilon=0.05, step=3, iteration=20, randomize=True):
    delta = torch.zeros_like(query).cuda()
    # if COCO:(-1, 1) else: (0, 1)

    noisy_output, _ = model(adv)
    per1, flag = probabiliby_mean_function(noisy_output, label_code, train_label, 1 - train_label)
    p = min((1 + 3 * per1) / 2, 1)

    if flag <= 0 and epoch > 10:
        iteration = math.ceil(iteration * p)
        epsilon = math.ceil(epsilon * p * 100) / 100
        if randomize:
            delta.uniform_(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(0, 1) - query.data

        delta.requires_grad = True
        for i in range(iteration):
            noisy_output, _ = model((query + delta))
            loss = loss_function(noisy_output, label_code, 1 - train_label, train_label, tau)
            loss.backward()

            delta.data = delta - step / 500 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(0, 1) - query.data
            delta.grad.zero_()
    else:
        if randomize:
            delta.uniform_(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(0, 1) - query.data

        delta.requires_grad = True
        for i in range(iteration):
            noisy_output, _ = model((query + delta))
            loss = loss_function(noisy_output, label_code, 1 - train_label, train_label, tau)
            loss.backward()

            delta.data = delta - step / 500 * torch.sign(delta.grad.detach())
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (query.data + delta.data).clamp(0, 1) - query.data
            delta.grad.zero_()

    return query + delta.detach()

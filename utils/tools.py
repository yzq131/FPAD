import random
import torch
import os
import errno
from torch.autograd import Variable
import numpy as np
from sklearn.cluster import KMeans
import skfuzzy as fuzz


def GenerateCode(model_img, model_txt, X, Y, num_data, bit):
    Bi = torch.zeros(num_data, bit)
    Bt = torch.zeros(num_data, bit)
    for i in range(num_data):
        data_img = Variable(X[i].type(torch.FloatTensor).cuda())
        data_text = Variable(Y[i].type(torch.FloatTensor).cuda())
        if data_img.dim() < 4:
            data_img = data_img.unsqueeze(0)
        if data_text.dim() < 2:
            data_text = data_text.unsqueeze(0)
        img_out, _ = model_img(data_img)
        text_out, _ = model_txt(data_text)
        Bi[i, :] = torch.sign(img_out.data.cpu())
        Bt[i, :] = torch.sign(text_out.data.cpu())
    return Bi, Bt


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_hamming(B1, B2):
    num = B1.shape[0]
    q = B1.shape[1]
    result = torch.zeros(num).cuda()
    for i in range(num):
        result[i] = 0.5 * (q - B1[i].dot(B2[i]))
    return result


def return_samples(index, qB, rB, k=None):
    num_query = qB.shape[0]
    if k is None:
        k = rB.shape[0]
    index_matrix = torch.zeros(num_query, k + 1).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat((index[i].unsqueeze(0), ind[np.linspace(0, rB.shape[0] - 1, k).astype('int')]), 0)
    return index_matrix


def return_results(index, qB, rB, s=None, o=None):
    num_query = qB.shape[0]
    index_matrix = torch.zeros(num_query, 1 + s + o).int()
    index = torch.from_numpy(index)
    for i in range(num_query):
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        index_matrix[i] = torch.cat(
            (index[i].unsqueeze(0), ind[:s], ind[np.linspace(0, rB.shape[0] - 1, o).astype('int')]), 0)
    return index_matrix


def CalcMap(qB, rB, query_label, retrieval_label, k=None):
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def CalcSim(batch_label, train_label):
    S = (batch_label.mm(train_label.t()) > 0).float()
    return S


def log_trick(x):
    lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    return lt


def image_normalization(_input):
    _input = 2 * _input / 255 - 1
    return _input


def image_restoration(_input):
    _input = (_input + 1) / 2 * 255
    return _input


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def run_fuzzy_cmeans(data, centroids_init, nclass):
    data_t = data.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_t, c=nclass, m=1.01, error=1e-7, maxiter=35, init=init_membership(data, nclass, centroids_init)
    )
    labels = np.argmax(u, axis=0)
    centers = np.zeros((nclass, data.shape[1]))
    for i in range(nclass):
        assigned_points = data[labels == i]
        if len(assigned_points) > 0:
            centers[i] = np.mean(assigned_points, axis=0)

    return cntr, u


def init_membership(data, n_clusters, centroids_init):
    n_samples = data.shape[0]

    kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=42, init=centroids_init).fit(data)
    labels = kmeans.labels_

    u0 = np.zeros((n_clusters, n_samples))

    for i in range(n_samples):
        u0[labels[i], i] = 1.0

    return u0


def calc_neighbor(label1, label2):
    Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    return Sim


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))

    loss = term1 + gamma * term2 + eta * term3
    return loss


def save_model(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    num_query = query_L.shape[0]
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def seed_setting(seed=64):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

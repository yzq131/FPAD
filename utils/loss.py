import torch


def loss_function(hash_codes, anchors, pos_mask, neg_mask, tau):
    eps = 1e-5
    bit = hash_codes.shape[1]
    batch_size = hash_codes.size(0)

    similarity = - 0.5 * (bit - hash_codes.mm(anchors.t()))

    pos_score = torch.exp(
        (similarity * pos_mask).sum(dim=1) / ((pos_mask.sum(dim=1) + eps) * bit) * tau
    )

    neg_score = (torch.exp(similarity * tau / bit) * neg_mask).sum(dim=1) + pos_score

    loss = -torch.log(pos_score / neg_score).sum() / batch_size

    return loss


def loss_cons_function(B1, B2, num_classes, omega, tau):
    eps = 1e-5
    bit = B1.shape[1]
    identity_mask = torch.eye(num_classes, device=B1.device)
    neg_mask = 1.0 - identity_mask

    sim_matrix = B1.mm(B2.t())
    batch = B1.shape[0]

    pos_term = torch.exp(
        (sim_matrix * identity_mask).sum(dim=1) / ((identity_mask.sum(dim=1) + eps) * bit) * tau
    )

    denominator = (torch.exp(sim_matrix * tau / bit) * neg_mask).sum(dim=1) + pos_term

    omega = torch.tensor(omega, dtype=pos_term.dtype, device=pos_term.device)
    loss = - (omega * torch.log(pos_term / denominator)).sum() / batch

    return loss


def probabiliby_mean_function(hash_codes, anchors, pos_mask, neg_mask):
    logits = hash_codes.mm(anchors.t())
    bit_len = hash_codes.size(1)
    batch = hash_codes.size(0)

    pos_exp = torch.exp((logits * pos_mask).sum(dim=1) / bit_len)
    total_exp = torch.exp(logits / bit_len).sum(dim=1)
    prob_avg = (pos_exp / total_exp).sum() / batch

    decision_flag = 0
    for i in range(batch):
        pos_vals = logits[i] * pos_mask[i]
        neg_vals = logits[i] * neg_mask[i]
        pos_vals = pos_vals[pos_vals != 0]
        neg_vals = neg_vals[neg_vals != 0]

        if (pos_vals.mean() - neg_vals.max()) > 0:
            decision_flag += 1
        else:
            decision_flag -= 1

    return prob_avg, decision_flag

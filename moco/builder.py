import torch
import torch.nn as nn
from functools import partial
# from torchvision.models import resnet
import torch.nn.functional as F
import pdb
import numpy as np

class ClaireNet(nn.Module):
    def __init__(self, 
        base_encoder=None, 
        in_dim=128,
        lat_dim=64,
        block_level=1,
        init='uniform', 
        args=None):

        super(ClaireNet, self).__init__()
        self.K = args.moco_k
        self.m = args.moco_m
        self.T = args.moco_t

        self.symmetric = args.symmetric
        self.batch_size = args.batch_size


        # create the encoders
        self.encoder_q = base_encoder(
                in_dim=in_dim,
                lat_dim=lat_dim,
                block_level=block_level,
                # init='uniform'      # does not matter
            )

        self.encoder_k = base_encoder(
                in_dim=in_dim,
                lat_dim=lat_dim,
                block_level=block_level
                # init='uniform'
            )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.zeros(lat_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        # pdb.set_trace()
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, im_q, im_k):
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1) # dot product
        # l_pos = torch.einsum('nc,ck->nk', [q, k.T])
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()]) # dot product
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T

        # # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        # pdb.set_trace()
        return loss, q, k


    def forward(self, weak1, weak2):
        """
        Input:
            weak1: a batch of query images
            weak2: a batch of key images
        Output:
            loss
        """
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        if self.symmetric:  # symmetric loss
            loss_12, q1, k2 = self.contrastive_loss(weak1, weak2)
            loss_21, q2, k1 = self.contrastive_loss(weak2, weak1)
            loss = (loss_12 + loss_21) * 0.5
            self._dequeue_and_enqueue(k1)
            self._dequeue_and_enqueue(k2)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(weak1, weak2)
            self._dequeue_and_enqueue(k)

        return loss
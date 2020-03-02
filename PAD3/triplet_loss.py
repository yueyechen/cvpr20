import torch
import torch.nn.functional as F
import numpy as np

#import mxnet as mx
import numpy as np
from easydict import EasyDict


def _pairwise_distances(config, feature, squared=False):
    """
    Computes the pairwise distance matrix with numerical stability.
  output[i, j] = || feature[i, :] - feature[j, :] ||_2
    :param feature:  2-D ndarray of size [number of data, feature dimension].
    :param squared: Boolean, whether or not to square the pairwise distances.

    :return: pairwise_distances: 2-D ndarray of size [number of data, number of data].
    """
    pairwise_distances_squared = torch.sum(feature * feature, dim=1, keepdim=True) + \
        torch.sum(torch.transpose(feature, 0, 1) * torch.transpose(feature, 0, 1), dim=0, keepdim=True) - \
                                 2.0 * torch.mm(feature, torch.transpose(feature, 0, 1))
                        
    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.max(pairwise_distances_squared, torch.zeros_like(pairwise_distances_squared))
    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, torch.zeros_like(pairwise_distances_squared))

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared +
                                        error_mask.to(torch.float32) * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = pairwise_distances * torch.logical_not(error_mask).to(torch.float32)

    # num_data = feature.shape_array[0]
    num_data = feature.size(0)
    # Explicitly set diagonals to zero.
    mask_offdiagonals = torch.ones_like(pairwise_distances) - torch.diag(
        torch.ones([num_data]).to(config.gpu))
    pairwise_distances = pairwise_distances * mask_offdiagonals
    return pairwise_distances


def masked_maximum(data, mask, dim=1):
    '''
    :param data:  2-D mx.nd.array of size [n,m] .
    :param mask:  2-D mx.nd.array of size [n,m] .
    :param dim:  the dimension over which to computer the maximum.
    :return:  masked_maximum: N-D mx.nd.array.
             the maximum dimension is of size 1 after the operation.
    '''
    axis_minimums = torch.min(data, dim, keepdim=True)[0]
    masked_maximums = torch.max(
        (data - axis_minimums) * mask, dim,
        keepdim=True)[0] + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    '''
    :param data:  2-D mx.nd.array of size [n,m] .
    :param mask:  2-D mx.nd.array of size [n,m] .
    :param dim:  the dimension over which to computer the maximum.
    :return:  masked_minimum: N-D mx.nd.array.
             the minimized dimension is of size 1 after the operation.
    '''
    axis_maximums = torch.max(data, dim, keepdim=True)[0]
    masked_minimums = torch.min(
        (data - axis_maximums) * mask, dim,
        keepdim=True)[0] + axis_maximums
    return masked_minimums


def triplet_semihard_loss(config, labels, embeddings, margin=1.0):
    """Computes the triplet loss with semi-hard negative mining.
     The loss encourages the positive distances (between a pair of embeddings with
     the same labels) to be smaller than the minimum negative distance among
     which are at least greater than the positive distance plus the margin constant
     (called semi-hard negative) in the mini-batch. If no such negative exists,
     uses the largest negative distance instead.
     See: https://arxiv.org/abs/1503.03832.
     Args:
       labels: 1-D mx.ndarray shape [batch_size] of
         multiclass integer labels.
       embeddings: 2-D ndarray of embedding vectors. Embeddings should
         be l2 normalized.
       margin: Float, margin term in the loss definition.
     Returns:
       triplet_loss: mx.nd.array float32 scalar.
     """
    # print(labels)
    lshape = labels.shape
    # assert len(lshape) == 1
    labels = torch.reshape(labels, [lshape[0], 1])

    # Build pairwise squared distance matrix.
    pdist_matrix = _pairwise_distances(config, embeddings, squared=True)
    # Build pairwise binary adjacency matrix.
    adjacency = (labels == torch.transpose(labels, 0, 1))
    #print(adjacency)
    # Invert so we can select negatives only.
    adjacency_not = torch.logical_not(adjacency)

    # batch_size = F.size_array(labels)
    batch_size = len(labels)

    # Compute the mask
    pdist_matrix_tile = pdist_matrix.repeat([batch_size, 1])
    mask = adjacency_not.repeat([batch_size, 1]) & \
        torch.gt(pdist_matrix_tile, torch.reshape( \
                torch.transpose(pdist_matrix, 0, 1), [-1, 1]))
    
    mask_final = torch.reshape(
        torch.gt(
            torch.sum(
                mask.to(torch.float32), 1, keepdim=True),
            0.0), [batch_size, batch_size])

    mask_final = torch.transpose(mask_final, 0, 1)

    adjacency_not = adjacency_not.to(torch.float32)
    mask = mask.to(torch.float32)

    # negatives_outside: smallest D_an where D_an > D_ap.
    negatives_outside = torch.reshape(
        masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
    negatives_outside = torch.transpose(negatives_outside, 0, 1)

    # negatives_inside: largest D_an.
    negatives_inside = masked_maximum(pdist_matrix, adjacency_not).repeat([1, batch_size])
    semi_hard_negatives = torch.where(
        mask_final, negatives_outside, negatives_inside)

    loss_mat = margin + (pdist_matrix - semi_hard_negatives)

    mask_positives = adjacency.to(torch.float32) - torch.diag(
        torch.ones([batch_size]).to(config.gpu))

    # In lifted-struct, the authors multiply 0.5 for upper triangular
    #   in semihard, they take all positive pairs except the diagonal.
    num_positives = torch.sum(mask_positives)

    triplet_loss = torch.sum(torch.max((loss_mat * mask_positives), torch.zeros_like(loss_mat * mask_positives))) / num_positives

    return triplet_loss



if __name__ == '__main__':
    # labels = mx.nd.array([1, 0, 1, 1, 0])
    # test_feature = mx.nd.array([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [2, 1, 4]], dtype='float32')
    # print(_pairwise_distances(test_feature))
    # print(triplet_semihard_loss(labels, test_feature))
    labels = torch.tensor([1, 0, 1, 1, 0])
    test_feature = torch.tensor([[1, 2, 3], [2, 3, 4], [1, 2, 3], [2, 3, 4], [2, 1, 4]], dtype=torch.float32)
    config = EasyDict()
    config.batch_size_per_gpu = 5
    print(_pairwise_distances(config, test_feature))
    print(triplet_semihard_loss(config, labels, test_feature))

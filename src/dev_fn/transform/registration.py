import torch

from .transform import assemble_T


def registration(src, dst):
    # src [B, N, D]
    # dst [B, N, D]

    # Centering
    src_center = torch.mean(src, dim=1, keepdim=True)
    dst_center = torch.mean(dst, dim=1, keepdim=True)

    src_centered = src - src_center
    dst_centered = dst - dst_center

    # Kabsch
    rotmat_inv, rmsd_value = kabsch_rmsd(src_centered, dst_centered)
    rotmat = rotmat_inv.transpose(2, 1)

    tsl = dst_center - torch.matmul(src_center, rotmat_inv)
    tsl = tsl.squeeze(1)

    transf = assemble_T(tsl, rotmat)
    return transf, rmsd_value


def kabsch_rmsd(P, Q):
    U = kabsch(P, Q)
    P = torch.matmul(P, U)
    return U, rmsd(P, Q)


def kabsch(P, Q):
    """
    The optimal rotation matrix U is calculated and then used to rotate matrix
    P unto matrix Q so the minimum root-mean-square deviation (RMSD) can be
    calculated.
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a translation of P and Q
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : tensor
        (B,N,D) matrix, where N is points and D is dimension.
    Q : tensor
        (B,N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : tensor
        Rotation matrix (B,D,D)
    Example
    -----
    """

    # Computation of the covariance matrix
    C = torch.matmul(P.transpose(1, 2), Q)  # [B, D, D]

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = torch.linalg.svd(C)
    # V [B, D, D], S [B, D], W [B, D, D]
    d = (torch.linalg.det(V) * torch.linalg.det(W) < 0.)  # [B, ]
    S[d, -1] = -S[d, -1]
    V[d, :, -1] = -V[d, :, -1]

    # Create Rotation matrix U
    U = torch.matmul(V, W)  # [B, D, D]
    return U


def rmsd(V, W):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.
    Parameters
    ----------
    V : tensor
        (B,N,D) matrix, where N is points and D is dimension.
    W : array
        (B,N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    rmsd : float
        Root-mean-square deviation
    """
    D = V.shape[-1]
    N = V.shape[-2]
    diff = V - W
    rmsd_value = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=(-2, -1)) / N)
    return rmsd_value

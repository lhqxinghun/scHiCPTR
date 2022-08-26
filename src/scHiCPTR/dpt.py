import logging
import numpy as np
from typing import Optional,Union
import scipy
from scipy.sparse import csr_matrix, issparse

# The code references https://github.com/theislab/scanpy

class DPT():
    """\
    """
    def __init__(self, connectivities):
        self.W = connectivities
        self.Z = None
        self._transitions_sym = None
        self._eigen_values = None
        self._eigen_basis = None
        self._dpt = None

    def compute_transitions(self, density_normalize: bool = True):
        """\
        """
        print("======= computing transitions ========") 
        if density_normalize:
            q = np.asarray(self.W.sum(axis=0))
            if not issparse(self.W):
                Q = np.diag(1.0/q)
            else:
                Q = scipy.sparse.spdiags(1.0/q, 0, self.W.shape[0], self.W.shape[0])
            K = Q @ self.W @ Q
        else:
            K = self.W

        z = np.sqrt(np.asarray(K.sum(axis=0)))
        if not issparse(K):
            self.Z = np.diag(1.0/z)
        else:
            self.Z = scipy.sparse.spdiags(1.0/z, 0, K.shape[0], K.shape[0])
        self._transitions_sym = self.Z @ K @ self.Z
        print("====== finished ======")  
    
    def compute_eigen(
        self,
        n_comps: int = 20,
        sym: Optional[bool] = None,
        sort: str = 'decrease',
    ):
        """\
        """
        np.set_printoptions(precision=10)
        if self._transitions_sym is None:
            raise ValueError('Run `.compute_transitions` first.')
        matrix = self._transitions_sym
        # compute the spectrum
        if n_comps == 0:
            evals, evecs = scipy.linalg.eigh(matrix)
        else:
            n_comps = min(matrix.shape[0]-1, n_comps)
            # ncv = max(2 * n_comps + 1, int(np.sqrt(matrix.shape[0])))
            ncv = None
            which = 'LM' if sort == 'decrease' else 'SM'
            # it pays off to increase the stability with a bit more precision
            matrix = matrix.astype(np.float64)
            evals, evecs = scipy.sparse.linalg.eigsh(
                matrix, k=n_comps, which=which, ncv=ncv
            )
            evals, evecs = evals.astype(np.float32), evecs.astype(np.float32)
        if sort == 'decrease':
            evals = evals[::-1]
            evecs = evecs[:, ::-1]
        logging.info(
            '    eigenvalues of transition matrix\n'
            '    {}'.format(str(evals).replace('\n', '\n    '))
        )
        # if self._number_connected_components > len(evals)/2:
        #     logg.warning('Transition matrix has many disconnected components!')
        self._eigen_values = evals
        self._eigen_basis = evecs

    def get_dpt_row(self, root: int=0):
        mask = None
        eigen_values = self._eigen_values
        eigen_basis = self._eigen_basis
        row = sum(
            (
                eigen_values[l] / (1-eigen_values[l])
                * (eigen_basis[root, l] - eigen_basis[:, l])
            )**2
        # account for float32 precision
            for l in range(0, eigen_values.size)
            if eigen_values[l] < 0.9994
        )
        # thanks to Marius Lange for pointing Alex to this:
        # we will likely remove the contributions from the stationary state below when making
        # backwards compat breaking changes, they originate from an early implementation in 2015
        # they never seem to have deteriorated results, but also other distance measures (see e.g.
        # PAGA paper) don't have it, which makes sense
        row += sum(
            (eigen_basis[root, l] - eigen_basis[:, l])**2
            for l in range(0, eigen_values.size)
            if eigen_values[l] >= 0.9994
        )
        if mask is not None:
            row[~mask] = np.inf
        _dpt = np.sqrt(row)
        self._dpt = _dpt / np.max(_dpt[_dpt < np.inf])

def compute_dpt(
    adj: Union[csr_matrix, None] = None,
    start: int = 0,
):
    dpt = DPT(adj)
    dpt.compute_transitions()
    dpt.compute_eigen()
    dpt.get_dpt_row(root=start)
    return dpt._dpt

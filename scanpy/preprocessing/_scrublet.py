import sys
from typing import Collection, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from anndata import AnnData

from . import _simple as pp
from . import _highly_variable_genes as hvg
from ._utils import _get_mean_var
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_array, sparsefuncs
from ..neighbors import neighbors

from .. import logging as logg

def scrublet(
    adata: AnnData, 
    sim_doublet_ratio: float = 2.0,
    n_neighbors: Optional[int] = None,
    expected_doublet_rate: float = 0.05, 
    find_highly_variable: bool = True,
    use_highly_variable: bool = True,
    pca_kwarg_dict: dict = {},
    gene_filter_kwarg_dict: dict = {},
    synthetic_doublet_umi_subsampling: float = 1.0, 
    log_transform: bool = False,
    scale: bool = True, 
    zero_center: bool = True,
    max_value: Optional[float] = None,
    copy: bool = False, 
    verbose: bool = True,
    knn_method: str = 'umap',
    knn_dist_metric: str = 'euclidean'
) -> Optional[AnnData]:
    """Predict doublets using Scrublet
    """

    adata_obs = adata.copy()

    if n_neighbors is None:
        n_neighbors = int(round(0.5*np.sqrt(adata.shape[0])))


    _print_optional('Preprocessing...', verbose)
    if 'n_counts' not in adata_obs.obs.keys():
        adata_obs.obs['n_counts'] = adata_obs.X.sum(1).A.squeeze()
    pp.normalize_per_cell(adata_obs, counts_per_cell_after=1e4)

    if use_highly_variable:
        if find_highly_variable:
            adata_obs.var['highly_variable'] = hvg.highly_variable_genes(
                pp.log1p(adata_obs, copy=True), 
                inplace=False,
                **gene_filter_kwarg_dict
            )['highly_variable']
        if 'highly_variable' in adata_obs.var.keys():
            adata_obs.raw = adata[:, adata_obs.var['highly_variable']]
            adata_obs = adata_obs[:, adata_obs.var['highly_variable']]
        else:
            raise ValueError('Did not find adata.var[\'highly_variable\']. '
                             'Please set `find_highly_variable=True`,  '
                             'set `use_highly_variable=False`, '
                             'or add adata.var[\'highly_variable\'].')
    else:
        adata_obs.raw = adata

    _print_optional(f'Using {adata_obs.shape[1]} highly variable genes.')

    _print_optional('Simulating doublets...', verbose)
    adata_sim = _simulate_doublets(adata_obs, sim_doublet_ratio, synthetic_doublet_umi_subsampling)
    pp.normalize_per_cell(adata_sim, counts_per_cell_after=1e4, counts_per_cell=adata_sim.obs['n_counts'].values)
    
    if log_transform:
        pp.log1p(adata_obs)
        pp.log1p(adata_sim)
    if scale:
        mean, var = _get_mean_var(adata_obs.X)
        if zero_center:
            if issparse(adata_obs.X):
                adata_obs.X = adata_obs.X.toarray()
                adata_sim.X = adata_sim.X.toarray()
        _scale_precomputed(adata_obs.X, mean, var, zero_center)
        _scale_precomputed(adata_sim.X, mean, var, zero_center)
    elif zero_center:
        if issparse(adata_obs.X):
            adata_obs.X = adata_obs.X.toarray()
            adata_sim.X = adata_sim.X.toarray()
        mean = adata_obs.X.mean(0)
        adata_obs.X -= mean
        adata_sim.X -= mean
    if max_value is not None: 
        adata_obs.X[adata_obs.X > max_value] = max_value
        adata_sim.X[adata_sim.X > max_value] = max_value

    _print_optional('Running dimensionality reduction...', verbose)
    pca_obs, pca_components = pp.pca(
        adata_obs.X,
        return_info=True, 
        zero_center=zero_center,
        **pca_kwarg_dict
    )[:2]
    if issparse(adata_sim.X):
        pca_sim = safe_sparse_dot(
            check_array(adata_sim.X, accept_sparse='csr'), 
            pca_components.T)
    else:
        pca_sim = np.dot(
            (adata_sim.X - adata_obs.X.mean(0)[None, :]), 
            pca_components.T)        

    _print_optional('Calculating doublet scores...', verbose)
    doublet_scores_obs, doublet_scores_sim = _nearest_neighbor_classifier(
        pca_obs, 
        pca_sim, 
        expected_doublet_rate, 
        n_neighbors=n_neighbors, 
        method=knn_method, 
        knn_dist_metric=knn_dist_metric
        )

    adata.obs['doublet_score'] = doublet_scores_obs
    adata.uns['scrublet'] = {}
    adata.uns['scrublet']['doublet_scores_sim'] = doublet_scores_sim
    adata.uns['scrublet']['doublet_parents'] = adata_sim.obsm['doublet_parents']
    adata.uns['scrublet']['pca_obs'] = pca_obs
    adata.uns['scrublet']['pca_sim'] = pca_sim
    adata.uns['scrublet']['parameters'] = {
        'expected_doublet_rate': expected_doublet_rate,
        'sim_doublet_ratio': sim_doublet_ratio,
        'n_neighbors': n_neighbors, 
        'log_transform': log_transform, 
        'scale': scale, 
        'zero_center': zero_center, 
        'max_value': max_value,
        'variable_genes': adata_obs.var_names.values.astype(str)
    }
    
    call_doublets(adata, verbose=verbose)
    return 

    # # put back into the adata object or return
    # if inplace:
    #     adata.X = bayesdata.values.transpose()
    # else:
    #     return bayesdata.values.transpose()

def _simulate_doublets(
    adata: AnnData, 
    sim_doublet_ratio: float = 2.0,
    synthetic_doublet_umi_subsampling: float = 1.0
) -> AnnData:

    ''' Simulate doublets by adding the counts of random observed transcriptome pairs.

    Arguments
    ---------
    sim_doublet_ratio : float, optional (default: None)
        Number of doublets to simulate relative to the number of observed 
        transcriptomes. If `None`, self.sim_doublet_ratio is used.

    synthetic_doublet_umi_subsampling : float, optional (defuault: 1.0) 
        Rate for sampling UMIs when creating synthetic doublets. If 1.0, 
        each doublet is created by simply adding the UMIs from two randomly 
        sampled observed transcriptomes. For values less than 1, the 
        UMI counts are added and then randomly sampled at the specified
        rate.

    Sets
    ----
    doublet_parents_
    '''

    n_obs = adata.shape[0]
    n_sim = int(n_obs * sim_doublet_ratio)
    doublet_parents = np.random.randint(0, n_obs, size=(n_sim, 2))
    
    X1 = adata.raw.X[doublet_parents[:,0],:]
    X2 = adata.raw.X[doublet_parents[:,1],:]
    tots1 = adata.obs['n_counts'][doublet_parents[:,0]].values
    tots2 = adata.obs['n_counts'][doublet_parents[:,1]].values
    if synthetic_doublet_umi_subsampling < 1:
        X_sim, total_counts_sim = _subsample_counts(X1+X2, synthetic_doublet_umi_subsampling, tots1+tots2)
    else:
        X_sim = X1 + X2
        total_counts_sim = tots1 + tots2
    adata_sim = AnnData(X_sim)
    adata_sim.obs['n_counts'] = total_counts_sim
    adata_sim.obsm['doublet_parents'] = doublet_parents
    return adata_sim


def _subsample_counts(X, rate, original_totals):
    if rate < 1:
        X.data = np.random.binomial(np.round(X.data).astype(int), rate)
        current_totals = X.sum(1).A.squeeze()
        unsampled_orig_totals = original_totals - current_totals
        unsampled_downsamp_totals = np.random.binomial(np.round(unsampled_orig_totals).astype(int), rate)
        final_downsamp_totals = current_totals + unsampled_downsamp_totals
    else:
        final_downsamp_totals = original_totals
    return X, final_downsamp_totals

def _scale_precomputed(X, column_means, column_vars, zero_center=True):
    scale = np.sqrt(column_vars)
    if zero_center:
        X -= column_means
        scale[scale == 0] = 1e-12
        X /= scale
    else:
        if issparse(X):
            sparsefuncs.inplace_column_scale(X, 1/scale)
        else:
            X /= scale

def _nearest_neighbor_classifier(pca_obs, pca_sim, expected_doublet_rate, n_neighbors=40, method='annoy', knn_dist_metric='euclidean'):
    pca_merged = np.vstack((pca_obs, pca_sim))
    adata_merged = AnnData(np.zeros((pca_merged.shape[0], 1)))
    adata_merged.obsm['X_pca'] = pca_merged
    n_obs = pca_obs.shape[0]
    n_sim = pca_sim.shape[0]
    doub_labels = np.concatenate((np.zeros(n_obs, dtype=int), 
                                  np.ones(n_sim, dtype=int)))

    
    # Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
    k_adj = int(round(n_neighbors * (1 + n_sim / float(n_obs))))
    
    # Find k_adj nearest neighbors
    if method == 'annoy':
        neigh = _get_knn_graph(adata_merged.obsm['X_pca'], k=k_adj, dist_metric=knn_dist_metric, approx=True, return_edges=False)
        n_sim_neigh = (neigh >= n_obs).sum(1)
        n_obs_neigh = (neigh < n_obs).sum(1)
    elif method == 'umap':
        neighbors(
           adata_merged,
           n_neighbors=k_adj,
           n_pcs=adata_merged.obsm['X_pca'].shape[1],
           use_rep='X_pca', 
           metric=knn_dist_metric)
        adjacency = adata_merged.uns['neighbors']['connectivities'] > 0
        n_sim_neigh = adjacency[:, n_obs:].sum(1).A.squeeze()
        n_obs_neigh = adjacency[:, :n_obs].sum(1).A.squeeze()

    # Calculate doublet score based on ratio of simulated cell neighbors vs. observed cell neighbors
    rho = expected_doublet_rate
    r = n_sim / float(n_obs)
    nd = n_sim_neigh.astype(float)
    ns = n_obs_neigh.astype(float)
    N = (nd + ns).astype(float)
    print(k_adj, N.min(), N.max())
    
    # Bayesian
    q=(nd+1)/(N+2)
    Ld = q*rho/r/(1-rho-q*(1-rho-rho/r))

    doublet_scores_obs = Ld[doub_labels == 0]
    doublet_scores_sim = Ld[doub_labels == 1]

    return doublet_scores_obs, doublet_scores_sim




def _get_knn_graph(X, k=5, dist_metric='euclidean', approx=False, return_edges=True):
    '''
    Build k-nearest-neighbor graph
    Return edge list and nearest neighbor matrix
    '''
    if approx:
        try:
            from annoy import AnnoyIndex
        except:
            approx = False
            print('Could not find library "annoy" for approx. nearest neighbor search')
    if approx:
        if dist_metric == 'cosine':
            dist_metric = 'angular'
        npc = X.shape[1]
        ncell = X.shape[0]
        annoy_index = AnnoyIndex(npc, metric=dist_metric)

        for i in range(ncell):
            annoy_index.add_item(i, list(X[i,:]))
        annoy_index.build(10) # 10 trees

        knn = []
        for iCell in range(ncell):
            knn.append(annoy_index.get_nns_by_item(iCell, k + 1)[1:])
        knn = np.array(knn, dtype=int)

    else:
        if dist_metric == 'cosine':
            nbrs = NearestNeighbors(n_neighbors=k, metric=dist_metric, algorithm='brute').fit(X)
        else:
            nbrs = NearestNeighbors(n_neighbors=k, metric=dist_metric).fit(X)
        knn = nbrs.kneighbors(return_distance=False)

    if return_edges:
        links = set([])
        for i in range(knn.shape[0]):
            for j in knn[i,:]:
                links.add(tuple(sorted((i,j))))
        return links, knn
    return knn

def _print_optional(string, verbose=True):
    if verbose:
        print(string)
    return

def call_doublets(adata, threshold=None, verbose=True):
    ''' Call trancriptomes as doublets or singlets

    Arguments
    ---------
    threshold : float, optional (default: None) 
        Doublet score threshold for calling a transcriptome
        a doublet. If `None`, this is set automatically by looking
        for the minimum between the two modes of the `doublet_scores_sim_`
        histogram. It is best practice to check the threshold visually
        using the `doublet_scores_sim_` histogram and/or based on 
        co-localization of predicted doublets in a 2-D embedding.

    verbose : bool, optional (default: True)
        If True, print summary statistics.

    Sets
    ----
    predicted_doublets_, z_scores_, threshold_,
    detected_doublet_rate_, detectable_doublet_fraction, 
    overall_doublet_rate_
    '''

    Ld_obs = adata.obs['doublet_score'].values
    Ld_sim = adata.uns['scrublet']['doublet_scores_sim']

    if threshold is None:
        # automatic threshold detection
        # http://scikit-image.org/docs/dev/api/skimage.filters.html


        try:
            from skimage.filters import threshold_minimum
        except ImportError:
            raise ImportError(
                'Please install the package scikit-image to enable automatic threshold calling.'
                'To set threshold manually, use `sc.pp.scrublet.call_doublets()`.')        
        try:
            threshold = threshold_minimum(Ld_sim)
            if verbose:
                print("Automatically set threshold at doublet score = {:.2f}".format(threshold))
        except:
            adata.obs['predicted_doublet'] = pd.Categorical(np.repeat(False, adata.obs.shape[0]))
            adata.uns['scrublet']['threshold'] = 1
            if verbose:
                print("Warning: failed to automatically identify doublet score threshold. Run `sc.pp.scrublet.call_doublets()` with user-specified threshold.")
            return 

    adata.uns['scrublet']['threshold'] = threshold
    adata.obs['predicted_doublet'] = pd.Categorical(Ld_obs > threshold)
    detected_rate = (Ld_obs > threshold).sum() / float(len(Ld_obs))
    detectable_frac = (Ld_sim > threshold).sum() / float(len(Ld_sim))

    adata.uns['scrublet']['detected_doublet_rate'] = detected_rate
    adata.uns['scrublet']['detectable_doublet_fraction'] = detectable_frac
    adata.uns['scrublet']['overall_doublet_rate'] = detected_rate / detectable_frac

    if verbose:
        print('Detected doublet rate = {:.1f}%'.format(100*detected_rate))
        print('Estimated detectable doublet fraction = {:.1f}%'.format(100 * detectable_frac))
        print('Overall doublet rate:')
        print('\tExpected   = {:.1f}%'.format(100 * adata.uns['scrublet']['parameters']['expected_doublet_rate']))
        print('\tEstimated  = {:.1f}%'.format(100 * adata.uns['scrublet']['overall_doublet_rate']))
        
    return

def plot_histogram(adata, scale_hist_obs='log', scale_hist_sim='linear', fig_size = (8,3)):
    ''' Plot histogram of doublet scores for observed transcriptomes and simulated doublets 

    The histogram for simulated doublets is useful for determining the correct doublet 
    score threshold. To set threshold to a new value, T, run call_doublets(threshold=T).

    '''

    import matplotlib.pyplot as plt
    threshold = adata.uns['scrublet']['threshold']
    fig, axs = plt.subplots(1, 2, figsize = fig_size)

    ax = axs[0]
    ax.hist(adata.obs['doublet_score'], np.linspace(0, 1, 50), color='gray', linewidth=0, density=True)
    ax.set_yscale(scale_hist_obs)
    yl = ax.get_ylim()
    ax.set_ylim(yl)
    ax.plot(threshold * np.ones(2), yl, c='black', linewidth=1)
    ax.set_title('Observed transcriptomes')
    ax.set_xlabel('Doublet score')
    ax.set_ylabel('Prob. density')

    ax = axs[1]
    ax.hist(adata.uns['scrublet']['doublet_scores_sim'], np.linspace(0, 1, 50), color='gray', linewidth=0, density=True)
    ax.set_yscale(scale_hist_sim)
    yl = ax.get_ylim()
    ax.set_ylim(yl)
    ax.plot(threshold * np.ones(2), yl, c = 'black', linewidth = 1)
    ax.set_title('Simulated doublets')
    ax.set_xlabel('Doublet score')
    ax.set_ylabel('Prob. density')

    fig.tight_layout()

    return fig, axs




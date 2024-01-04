"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-12-15
Desc: This script used PCA techinique to extract
PCs from subcellular localization data.
"""

from sklearn.decomposition import PCA
import numpy as np
import os

if __name__ == '__main__':
    print("=====>Load subcellular localization data...")
    subcellular = np.load("test_data/subcellular.npy")
    pc = 256
    print("=====>Select the top 256 PCs...")
    pca = PCA(n_components=pc)
    X_pca = pca.fit_transform(subcellular)
    np.save('data/pca_sub.npy', X_pca)
    print("=====>Processed data are stored under data/PCA_sub.npy")
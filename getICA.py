"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-12-15
Desc: This script used PCA techinique to extract
ICs from subcellular localization data.
"""

from sklearn.decomposition import FastICA
import numpy as np

if __name__ == '__main__':
    print("=====>Load subcellular localization data...")
    subcellular = np.load("test_data/subcellular.npy")
    num_features_to_select = 256
    print("=====>Select the top 256 ICs...")
    ica = FastICA(n_components=num_features_to_select)
    S_ = ica.fit_transform(subcellular)
    np.save('data/ICA_sub.npy', S_)
    print("=====>Processed data are stored under data/ica_sub.npy")


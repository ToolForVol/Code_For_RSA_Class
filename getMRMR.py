"""
Author: Ye Chen ^(=•ェ•=)^
Date: 2023-12-15
Desc: This script used mRMR techinique to extract
features from subcellular localization data.
"""


from skfeature.function.information_theoretical_based import MRMR
from tqdm.notebook import trange
import numpy as np
import os

if __name__ == '__main__':
    print("=====>Load subcellular localization data...")
    subcellular = np.load("test_data/subcellular.npy")
    label = np.load(tempPath + '/标签/label.npy')
    num_features_to_select = 256
    print("=====>Select top 256 features...")
    s_feature, jcmi, mi = MRMR.mrmr(subcellular, label, n_selected_features=num_features_to_select)
    np.save("data/mRMR_sub.npy", subcellular[:, s_feature])
    print("=====>Processed data are stored under data/mRMR_sub.npy")
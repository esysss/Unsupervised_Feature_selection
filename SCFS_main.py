
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



import numpy as np
import scipy.io
from SCFS import scfs
from tqdm import tqdm
import clus

from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import adjusted_rand_score

# Please cite "Unsupervised Feature Selection based on Adaptive Similarity Learning and Subspace Clustering", Mohsen Ghassemi Parsa, Hadi Zare, Mehdi Ghatee

data_name = 'Datasets/lung.mat'
print (data_name)

mat = scipy.io.loadmat(data_name)
X = mat['X']
X = X.astype(float)
X += np.fabs(np.min(X))
y = mat['Y']
y = y[:, 0]
Parm = [1e-4, 1e-2, 1, 1e+2, 1e+4]

n, p = X.shape
c = len(np.unique(y))

XX = np.dot(X, np.transpose(X))
XTX = np.dot(np.transpose(X), X)

count = 0

idx = np.zeros((p, 25), dtype=np.int)

sample_size = [50,100,150,200,250,300]
kmeans_repeats = 20

VM = {50:0, 100:0, 150:0, 200:0, 250:0, 300:0}
NMI = {50:0, 100:0, 150:0, 200:0, 250:0, 300:0}
RS = {50:0, 100:0, 150:0, 200:0, 250:0, 300:0}

for Parm1 in Parm:
    for Parm2 in Parm:
        W = scfs(X, XX=XX, XTX=XTX, n_clusters=c, alpha=Parm1, beta=Parm2)
        T = (W * W).sum(1)
        index = np.argsort(T, 0)
        idx[0:p, count] = index[::-1]
        count += 1

        # print('alpha= {}, beta= {}'.format(Parm1, Parm2))

        all_vmeasures = []
        all_nmis = []
        all_randscores = []

        index = index[::-1]

        nmitemp = []
        for i in sample_size:
            # print('number of selecting features= ',i)
            x = X[:,index[:i]]

            # print('for kmeans ...')
            yhats = [clus.kmeans(x,c) for i in range(kmeans_repeats)]
            vmeasures = [v_measure_score(y,yhats[i]) for i in range(kmeans_repeats)]
            nmis = [nmi(y, yhats[i], average_method='max') for i in range(kmeans_repeats)]
            randscores = [adjusted_rand_score(y, yhats[i]) for i in range(kmeans_repeats)]

            # print("sample size: {}, alpha: {}, beta: {}, NMI: {}".format(
            #     i, Parm1, Parm2, np.max(nmis)
            # ))
            if VM[i] < np.mean(vmeasures):
                VM[i] = np.mean(vmeasures)

            if NMI[i] < np.mean(nmis):
                NMI[i] = np.mean(nmis)

            if RS[i] < np.mean(randscores):
                RS[i] = np.mean(randscores)

            nmitemp.append(np.mean(nmis))

        print("NMI: {}, ALPHA: {}, BETA:{}".format(np.mean(nmitemp), Parm1, Parm2))




print('******************** all done *******************')
print('V-measure: {} (mean) {} (std)'.format(np.mean(list(VM.values())), np.std(list(VM.values()))))
print('NMI: {} (mean) {} (std)'.format(np.mean(list(NMI.values())), np.std(list(NMI.values()))))
print('Rand_Score: {} (mean) {} (std)'.format(np.mean(list(RS.values())), np.std(list(RS.values()))))

print("V-measure: {}\nNMI: {}\nRand_Score: {}".format(VM,NMI,RS))
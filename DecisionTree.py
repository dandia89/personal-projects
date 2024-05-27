import numpy as np

class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
    def build_tree(self, Xtrain, Ytrain):
        def build_leaf(y_arr):
            yb = np.array([[-1, np.mean(y_arr), np.nan, np.nan]])
            return yb

        if Xtrain.shape[0] <= self.leaf_size or len(np.unique(Ytrain)) == 1:
            return build_leaf(Ytrain)
        index = self.split_feature(Xtrain, Ytrain)
        value = np.median(Xtrain[:, index])
        left_data_indices = Xtrain[:, index] <= value
        right_data_indices = Xtrain[:, index] > value

        edge = np.unique(left_data_indices)
        if edge.size == 1:
            return build_leaf(Ytrain[left_data_indices == edge.item()])
        edge = np.unique(right_data_indices)
        if edge.size == 1:
            return build_leaf(Ytrain[right_data_indices == edge.item()])

        left_subtree = self.build_tree(Xtrain[left_data_indices], Ytrain[left_data_indices])
        right_subtree = self.build_tree(Xtrain[right_data_indices], Ytrain[right_data_indices])
        root = np.array([[index, value, 1, left_subtree.shape[0]+1]])

        self.tree = np.concatenate((root, left_subtree, right_subtree), axis=0)

        return self.tree


    def split_feature(self, Xtrain, Ytrain):
        correlation_coef = np.corrcoef(Xtrain, Ytrain, rowvar=False)
        correlation_with_YTrain = correlation_coef[:-1,-1]
        max_corr_index = np.argmax(np.abs(correlation_with_YTrain))
        return max_corr_index
        
    def add_evidence(self, Xtrain, Ytrain):
        self.tree = self.build_tree(Xtrain,Ytrain)
    def query(self, Xtest):
        Ypred_array = np.zeros(Xtest.shape[0])
        for i in range(Xtest.shape[0]):
            j = 0
            while j < self.tree.shape[0]:
                if self.tree[j, 0] == -1:
                    Ypred = self.tree[j, 1]
                    Ypred_array[i] = Ypred
                    break
                else:
                    index_val = int(self.tree[j, 0])
                    split_val = float(self.tree[j, 1])
                    if Xtest[i, index_val] > split_val:
                        j = int(self.tree[j, 3]) + j
                    else:
                        j = j + 1
        return Ypred_array

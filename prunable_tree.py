from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import TREE_LEAF, TREE_UNDEFINED

class PrunableTree(DecisionTreeClassifier):
    def __init__(self, prune, **kwargs):
        super(PrunableTree, self).__init__(**kwargs)
        self.prune = prune
        self.inner_nodes_prev = 0
        self.inner_nodes_post = 0
    
    def fit(self, *args, **kwargs):
        super(PrunableTree, self).fit(*args, **kwargs)
        self.inner_nodes_prev = sum(self.tree_.children_left != TREE_LEAF)
        if self.prune:
            threshold = max(self.tree_.impurity) / 2
            prune(self.tree_, 0, threshold)
        self.inner_nodes_post = sum(self.tree_.children_left != TREE_LEAF)

def prune(tree, index, threshold):
    if tree.children_left[index] != tree.children_right[index]:
        prune(tree, tree.children_left[index], threshold)
        prune(tree, tree.children_right[index], threshold)

        if tree.impurity[index] < threshold:
            make_leaf(tree, index)

def make_leaf(tree, index):
    tree.children_left[index] = TREE_LEAF
    tree.children_right[index] = TREE_LEAF
    tree.feature[index] = TREE_UNDEFINED
    tree.threshold[index] = TREE_UNDEFINED


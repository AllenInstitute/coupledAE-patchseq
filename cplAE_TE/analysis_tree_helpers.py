import fnmatch
import os
import pprint

import feather
import numpy as np
import pandas as pd
import scipy.io as sio
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

class Node():
    '''Simple Node class. Each instance contains a list of children and parents.'''

    def __init__(self,name,C_list=[],P_list=[]):
        self.name=name
        self.C_name_list = C_list[P_list==name]
        self.P_name = P_list[C_list==name]
        return
    def __repr__(self):
        #Invoked when printing a list of Node objects
        return self.name
    def __str__(self):
        #Invoked when printing a single Node object
        return self.name
    def __eq__(self,other):
        
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False
    def children(self,C_list=[],P_list=[]):
        return [Node(n,C_list,P_list) for n in self.C_name_list]

def get_valid_classifications(current_node_list,C_list,P_list,valid_classes):
    '''Recursively generates all possible classifications that are valid, 
    based on the hierarchical tree defined by `C_list` and `P_list` \n
    `current_node_list` is a list of Node objects. It is initialized as a list with only the root Node.'''
    
    current_node_list.sort(key=lambda x: x.name)
    valid_classes.append(sorted([node.name for node in current_node_list]))
    for node in current_node_list:
        current_node_list_copy = current_node_list.copy()
        children_node_list = node.children(C_list=C_list,P_list=P_list)
        if len(children_node_list)>0:
            current_node_list_copy.remove(node)
            current_node_list_copy.extend(children_node_list)
            if sorted([node.name for node in current_node_list_copy]) not in valid_classes:
                valid_classes = get_valid_classifications(current_node_list_copy,C_list=C_list,P_list=P_list,valid_classes=valid_classes)
    return valid_classes


class HTree():
    '''Class to work with hierarchical tree .csv generated for the transcriptomic data.
    `htree_file` is full path to a .csv. The original .csv was generated from dend.RData, 
    processed with `dend_functions.R` and `dend_parents.R` (Ref. Rohan/Zizhen)'''
    def __init__(self,htree_df=None,htree_file=None):
        
        #Load and rename columns from filename
        if htree_file is not None:
            htree_df = pd.read_csv(htree_file)
            htree_df = htree_df[['x', 'y', 'leaf', 'label', 'parent', 'col']]
            htree_df = htree_df.rename(columns={'label': 'child','leaf': 'isleaf'})

            #Sanitize values
            htree_df['isleaf'].fillna(False,inplace=True)
            htree_df['y'].values[htree_df['isleaf'].values] = 0.0
            htree_df['col'].fillna('#000000',inplace=True)
            htree_df['parent'].fillna('root',inplace=True)

            #Sorting for convenience
            htree_df = htree_df.sort_values(by=['y', 'x'], axis=0, ascending=[True, True]).copy(deep=True)
            htree_df = htree_df.reset_index(drop=True).copy(deep=True)
        
        #Set class attributes using dataframe columns
        for c in htree_df.columns:
            setattr(self, c, htree_df[c].values)
        return
    
    def obj2df(self):
        '''Convert HTree object to a pandas dataframe'''
        htree_df = pd.DataFrame({key:val for (key,val) in self.__dict__.items()})
        return htree_df
    
    def df2obj(self,htree_df):
        '''Convert a valid pandas dataframe to a HTree object'''
        for key in htree_df.columns:
            setattr(self, key, htree_df[key].values)
        return

    def plot(self,figsize=(15,10),fontsize=10,skeletononly=False,skeletoncol='#BBBBBB',skeletonalpha=1.0,ls='-',txtleafonly=False,fig=None):
        if fig is None:
            fig = plt.figure(figsize=figsize)

        #Labels are shown only for children nodes
        if skeletononly==False:
            if txtleafonly==False:
                for i, label in enumerate(self.child):
                        t = plt.text(self.x[i], self.y[i], label, 
                                color=self.col[i],
                                horizontalalignment='center',
                                verticalalignment='top', 
                                rotation=90, 
                                fontsize=fontsize)
            else:
                for i in np.flatnonzero(self.isleaf):
                    label = self.child[i]
                    plt.text(self.x[i], self.y[i], label, 
                            color=self.col[i],
                            horizontalalignment='center',
                            verticalalignment='top', 
                            rotation=90, 
                            fontsize=fontsize)
                    #t.set_bbox(dict(facecolor=self.col[i], edgecolor='w'))

        for parent in np.unique(self.parent):
            #Get position of the parent node:
            p_ind = np.flatnonzero(self.child==parent)
            if p_ind.size==0: #Enters here for any root node
                p_ind = np.flatnonzero(self.parent==parent)
                xp = self.x[p_ind]
                yp = 1.1*np.max(self.y)
            else:    
                xp = self.x[p_ind]
                yp = self.y[p_ind]

            all_c_inds = np.flatnonzero(np.isin(self.parent,parent))
            for c_ind in all_c_inds:
                xc = self.x[c_ind]
                yc = self.y[c_ind]
                plt.plot([xc, xc], [yc, yp], color=skeletoncol,alpha=skeletonalpha,ls=ls,)
                plt.plot([xc, xp], [yp, yp], color=skeletoncol,alpha=skeletonalpha,ls=ls)
        if skeletononly==False:
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim([np.min(self.x) - 1, np.max(self.x) + 1])
            ax.set_ylim([np.min(self.y), 1.2*np.max(self.y)])
            plt.tight_layout()
            fig.subplots_adjust(bottom=0.2)
        return
    
    def plotnodes(self,nodelist,fig=None):
        ind = np.isin(self.child,nodelist)
        plt.plot(self.x[ind], self.y[ind],'s',color='r')
        return

    def get_descendants(self,node:str,leafonly=False):
        '''Return a list consisting of all descendents for a given node. Given node is excluded.\n
        'node' is of type str \n
        `leafonly=True` returns only leaf node descendants'''

        descendants = []
        current_node = self.child[self.parent == node].tolist()
        descendants.extend(current_node)
        while current_node:
            parent = current_node.pop(0)
            next_node = self.child[self.parent == parent].tolist()
            current_node.extend(next_node)
            descendants.extend(next_node)
        if leafonly:
            descendants = list(set(descendants) & set(self.child[self.isleaf]))
        return descendants

    def get_all_descendants(self,leafonly=False):
        '''Return a dict consisting of node names as keys and, corresp. descendant list as values.\n
        `leafonly=True` returns only leaf node descendants'''
        descendant_dict = {}
        for key in np.unique(np.concatenate([self.child,self.parent])):
            descendant_dict[key]=self.get_descendants(node=key,leafonly=leafonly)
        return descendant_dict

    def get_ancestors(self,node,rootnode=None):
        '''Return a list consisting of all ancestors 
        (till `rootnode` if provided) for a given node.'''

        ancestors = []
        current_node = node
        while current_node:
            current_node = self.parent[self.child == current_node]
            ancestors.extend(current_node)
            if current_node==rootnode:
                current_node=[]
        return ancestors

    def get_mergeseq(self):
        '''Returns `ordered_merges` consisting of \n
        1. list of children to merge \n
        2. parent label to merge the children into \n
        3. number of remaining nodes in the tree'''

        # Log changes for every merge step
        ordered_merge_parents = np.setdiff1d(self.parent,self.child[self.isleaf])
        y = []
        for label in ordered_merge_parents:
            if np.isin(label,self.child):
                y.extend(self.y[self.child==label])
            else:
                y.extend([np.max(self.y)+0.1])
        
        #Lowest value is merged first
        ind = np.argsort(y)
        ordered_merge_parents = ordered_merge_parents[ind].tolist()
        ordered_merges = []
        while len(ordered_merge_parents) > 1:
            # Best merger based on sorted list
            parent = ordered_merge_parents.pop(0)
            children = self.child[self.parent == parent].tolist()
            ordered_merges.append([children, parent])
        return ordered_merges

    def get_subtree(self, node):
        '''Return a subtree from the current tree'''
        subtree_node_list = self.get_descendants(node=node)+[node]
        if len(subtree_node_list)>1:
            subtree_df = self.obj2df()
            subtree_df = subtree_df[subtree_df['child'].isin(subtree_node_list)]
        else:
            print('Node not found in current tree')
        return HTree(htree_df=subtree_df)

    def update_layout(self):
        '''Update `x` positions of tree based on newly assigned leaf nodes.
        '''
        #Update x position for leaf nodes to evenly distribute them.
        all_child = self.child[self.isleaf]
        all_child_x = self.x[self.isleaf]
        sortind = np.argsort(all_child_x)
        new_x = 0
        for (this_child,this_x) in zip(all_child[sortind],all_child_x[sortind]):
            self.x[self.child==this_child]=new_x
            new_x = new_x+1
            
            
        parents = self.child[~self.isleaf].tolist() 
        for node in parents:
            descendant_leaf_nodes = self.get_descendants(node=node,leafonly=True)
            parent_ind = np.isin(self.child,[node])
            descendant_leaf_ind = np.isin(self.child,descendant_leaf_nodes)
            self.x[parent_ind] = np.mean(self.x[descendant_leaf_ind])
        return 


def do_merges(labels, list_changes=[], n_merges=0, verbose=False):
    """Perform n_merges on an array of labels using the list of changes at each merge. 
    If labels are leaf node labels, then the do_merges() gives successive horizontal cuts of the hierarchical tree.
    
    Arguments:
        labels -- label array to update
    
    Keyword Arguments:
        list_changes  -- output of Htree.get_mergeseq()
        n_merges -- int, can be at most len(list_changes)
    
    Returns:
        labels -- array of updated labels. Same size as input, non-unique entries are allowed.
    """
    assert isinstance(labels,np.ndarray), 'labels must be a numpy array'
    for i in range(n_merges):
        if i < len(list_changes):
            c_nodes_list = list_changes[i][0]
            p_node = list_changes[i][1]
            for c_node in c_nodes_list:
                n_samples = np.sum([labels == c_node])
                labels[labels == c_node] = p_node
                if verbose:
                    print(n_samples,' in ',c_node, ' --> ' ,p_node)
        else:
            print('Exiting after performing max allowed merges =',
                  len(list_changes))
            break
    return labels


def simplify_tree(pruned_subtree,skip_nodes=None):
    '''pruned subtree has nodes that have a single child node. In the returned simplified tree,
    the parent is directly connected to the child, and such intermediate nodes are removed.'''
    
    simple_tree = deepcopy(pruned_subtree)
    if skip_nodes is None:
        X = pd.Series(pruned_subtree.parent).value_counts().to_frame()
        skip_nodes = X.iloc[X[0].values==1].index.values.tolist()

    for node in skip_nodes:
        node_parent = np.unique(simple_tree.parent[simple_tree.child == node])
        node_child = np.unique(simple_tree.child[simple_tree.parent == node])

        #Ignore root node special case:
        if node_parent.size != 0:
            #print(simple_tree.obj2df().to_string())
            print('Remove {} and link {} to {}'.format(node, node_parent, node_child))
            simple_tree.parent[simple_tree.parent == node]=node_parent

            #Remove rows containing this particular node as parent or child
            simple_tree_df=simple_tree.obj2df()
            simple_tree_df.drop(simple_tree_df[(simple_tree_df.child == node) | (simple_tree_df.parent == node)].index, inplace=True)

            #Reinitialize tree from the dataframe
            simple_tree=HTree(htree_df=simple_tree_df)
        
    return simple_tree,skip_nodes
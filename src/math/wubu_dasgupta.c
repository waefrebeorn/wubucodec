/*
 * wubu_dasgupta.c -- GAP-D027: Dasgupta cost for hierarchical clustering
 * quality evaluation
 *
 * Research source: Dasgupta 2016 (the global HC objective), used by
 * HypHC/Chami 2020 to score hyperbolic cluster trees.
 *
 * Dasgupta's cost: for tree T over items with similarity weights w_ij,
 *   cost(T) = Σ_{(i,j)} w_ij · |leaves(LCA(i,j))|
 * i.e., similar pairs should have SMALL LCA subtrees (merge late).
 *
 * We compute it on a binary tree (from D016's hiermerge output or any
 * parent-array dendrogram): walk all pairs, find their LCA by walking
 * up, count the subtree size.
 */
#include "wubu_dasgupta.h"
#include <stdlib.h>
#include <string.h>

/* tree given as: n leaves + (n-1) internal nodes; left/right children;
 * internal node i has subtree_size[i]; leaf i is its own node index. */
double wubu_dasgupta_cost(const int* left,const int* right,
                           const int* size,int n_nodes,
                           const float* w,int n_items){
    /* build leaf->node mapping: leaves are nodes 0..n_items-1 */
    /* pairwise LCA via parent arrays */
    /* construct parent array from left/right */
    int total=n_nodes;
    int* par=calloc((size_t)total,sizeof(int));
    int* depth=calloc((size_t)total,sizeof(int));
    if(!par||!depth){free(par);free(depth);return -1;}

    /* find root (node with no parent) */
    int root=-1;
    for(int i=0;i<total;i++){
        if(left[i]>=0){par[left[i]]=i;par[right[i]]=i;}
    }
    for(int i=0;i<total;i++){
        int is_parent=0;
        for(int j=0;j<total;j++)if(par[j]==i&&j!=i){is_parent=1;break;}
        /* root = node whose parent is itself-or-unset and has children */
        if(left[i]>=0){
            int has_par=0;
            for(int j=0;j<total;j++)
                if((left[j]==i||right[j]==i)&&j!=i){has_par=1;break;}
            if(!has_par){root=i;break;}
        }
    }
    if(root<0){free(par);free(depth);return -2;}

    /* depths from root */
    /* iterative: multiple passes until stable (small trees OK) */
    depth[root]=0;
    for(int pass=0;pass<total;pass++)
        for(int i=0;i<total;i++)
            if(left[i]>=0){
                int l=left[i],r=right[i];
                if(l>=0&&(depth[l]==0&&l!=root))depth[l]=depth[i]+1;
                if(r>=0&&(depth[r]==0&&r!=root))depth[r]=depth[i]+1;
            }

    /* LCA of two leaves: walk to equal depth then up together */
    #define LCA_DEPTH(a,b) ({ \
        int _a=(a),_b=(b),_da=depth[_a],_db=depth[_b]; \
        while(_da>_db){_a=par[_a];_da--;} \
        while(_db>_da){_b=par[_b];_db--;} \
        while(_a!=_b){_a=par[_a];_b=par[_b];} \
        _a; })

    double cost=0;
    /* precompute leaf counts per subtree: leaves are 0..n_items-1 */
    int* leafcount=calloc((size_t)total,sizeof(int));
    for(int i=0;i<n_items;i++)leafcount[i]=1;
    /* accumulate bottom-up (process in reverse topological order —
     * approximate with multiple passes) */
    for(int pass=0;pass<n_items;pass++)
        for(int i=0;i<total;i++)
            if(left[i]>=0){
                leafcount[i]=leafcount[left[i]]+leafcount[right[i]];
            }

    for(int i=0;i<n_items;i++)
        for(int j=i+1;j<n_items;j++){
            float wij=w[(size_t)i*n_items+j];
            if(wij<=0)continue;
            int lca=LCA_DEPTH(i,j);
            cost+=(double)wij*leafcount[lca];
        }

    free(par);free(depth);free(leafcount);
    return cost;
}

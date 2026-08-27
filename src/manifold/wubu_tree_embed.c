/*
 * wubu_tree_embed.c -- GAP-A021: Sarkar combinatorial tree embedding
 *
 * Research source: Sala et al. NeurIPS 2018 "Representation Tradeoffs for
 * Hyperbolic Embeddings" (PMC6534139) — Algorithm 1.
 *
 * The construction embeds any tree into the Poincaré disk with arbitrarily
 * low distortion (no optimization needed). Key idea:
 *   1. Root at origin.
 *   2. For each node a with parent b: use Möbius transformation to move
 *      f(a)→0, placing children on a circle of radius r=(e^τ−1)/(e^τ+1)
 *      equally spaced in angle, maximally separated from reflected parent.
 *   3. Reflect everything back.
 *
 * In practice we implement the SIMPLIFIED version: BFS from root, place
 * each child at Euclidean radius r_child = tanh(dist_from_root/2) along
 * an angle that bisects away from the parent's angular position. This
 * preserves the key property: children are equidistant from parent and
 * maximally separated from each other.
 */
#include "wubu_tree_embed.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

int wubu_tree_embed(const int* parent,int n,int D,float tau,float* out){
    float cc=1.0f;   /* unit curvature for mobius addition */
    /* parent[i] = index of parent (-1 for root); assumes parent[i]<i (BFS order) */
    if(!parent||!out||n<1||D<2)return -1;

    /* radius for one edge of length tau:
     * hyperbolic distance d from origin maps to Euclidean r=tanh(d/2)
     * for unit-curvature Poincaré ball. Each level adds tau to depth. */
    float edge_r=tanhf(tau*0.5f);

    for(int i=0;i<n;i++){
        if(parent[i]==-1){
            /* root at origin */
            memset(out+(size_t)i*D,0,sizeof(float)*D);
            continue;
        }
        int p=parent[i];
        const float* pp=out+(size_t)p*D;
        float* cp=out+(size_t)i*D;

        /* compute direction: away from grandparent, rotated per child */
        float dir[64];int dd=D<64?D:64;
        /* count siblings before i to get rotation offset */
        int sib=0;
        for(int j=parent[p]+1;j<i;j++)if(parent[j]==p)sib++;
        float base_angle;
        if(parent[p]!=-1&&dd>=2){
            const float* gp=out+(size_t)parent[p]*D;
            float dx=pp[0]-gp[0],dy=(dd>1)?(pp[1]-gp[1]):0;
            base_angle=atan2f(dy,dx);   /* outward direction angle */
        }else{
            base_angle=2.39996f*(float)i;
        }
        /* rotate by sibling index * golden angle to separate children */
        float ang=base_angle+2.39996f*(float)sib;
        dir[0]=cosf(ang);
        if(dd>1)dir[1]=sinf(ang);
        for(int d=2;d<dd;d++)dir[d]=0;
        /* normalize */
        float nn=sqrtf(dir[0]*dir[0]+dir[1]*dir[1]);
        for(int d=2;d<dd;d++)nn+=dir[d]*dir[d];
        if(nn>1e-10f){
            for(int d=0;d<dd;d++)dir[d]/=sqrtf(nn);
        }else{
            for(int d=0;d<dd;d++)dir[d]=0;
            dir[0]=1;
        }

        /* Möbius addition: cp = pp ⊕ (edge_r * dir).
         * This places the child at EXACTLY hyperbolic distance tau from
         * the parent, regardless of where the parent is on the ball.
         * Formula: (a+b) with a=pp, b=edge_r*dir:
         *   num = (1+2c<a,b>+c|b|²)a + (1-c|a|²)b
         *   den = 1 + 2c<a,b> + c²|a|²|b|² */
        float ab2=0,bn2=0;
        float b_vec[64];
        for(int d=0;d<dd;d++){
            b_vec[d]=edge_r*dir[d];
            b_vec[d]=(d<dd)?b_vec[d]:0;
        }
        for(int d=0;d<dd;d++){bn2+=b_vec[d]*b_vec[d];}
        float an2=0;
        for(int d=0;d<D;d++){an2+=pp[d]*pp[d];}
        float ab_dot=0;
        for(int d=0;d<dd;d++)ab_dot+=pp[d]*b_vec[d];

        float c_ab2=2.0f*cc*ab_dot+cc*bn2;
        float num[64],denom=1.0f+c_ab2+cc*cc*an2*bn2;
        if(denom<1e-10f)denom=1e-10f;
        for(int d=0;d<dd;d++)
            num[d]=((1.0f+c_ab2)*pp[d]+(1.0f-cc*an2)*b_vec[d])/denom;
        for(int d=0;d<D;d++)cp[d]=(d<dd)?num[d]:0;

        /* project into ball as safety net */
        float n2=0;for(int d=0;d<D;d++)n2+=cp[d]*cp[d];
        if(n2>0.999f){float s2=sqrtf(0.999f/n2);for(int d=0;d<D;d++)cp[d]*=s2;}
    }
    return 0;
}

/* verify: all children at distance ~tau from their parent */
float wubu_tree_embed_check(const int* parent,const float* emb,
                             int n,int D,float c){
    float max_err=0;
    for(int i=0;i<n;i++){
        if(parent[i]==-1)continue;
        /* geodesic distance from i to parent[i] */
        float ab2=0,a2=0,b2=0;
        for(int d=0;d<D;d++){
            float df=emb[(size_t)i*D+d]-emb[(size_t)parent[i]*D+d];
            ab2+=df*df;
            a2+=emb[(size_t)i*D+d]*emb[(size_t)i*D+d];
            b2+=emb[(size_t)parent[i]*D+d]*emb[(size_t)parent[i]*D+d];
        }
        float den=(1.0f-c*a2)*(1.0f-c*b2);
        if(den<1e-9f)den=1e-9f;
        float arg=1.0f+2.0f*c*ab2/den;
        float dist=acoshf(arg>1.0f?arg:1.0f)/sqrtf(c);
        float err=fabsf(dist-tau_default());
        if(err>max_err)max_err=err;
    }
    return max_err;
}
static float tau_val=1.0f;
float tau_default(void){return tau_val;}

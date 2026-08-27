/* test_wubu_knngraph.c -- GAP-A007 gates
 *  G1 every node has exactly k neighbors (unidirectional mode)
 *  G2 mutual mode: edges are symmetric (i→j iff j→i)
 *  G3 nearest neighbor is truly the geodesically-closest point
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include "wubu_knngraph.h"

static int passed=0,failed=0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL: %s (line %d)\n",#c,__LINE__); failed++; return; } }while(0)
static float kg_d(const float*a,const float*b,int D,float c){
    float ab2=0,a2=0,b2=0;
    for(int d=0;d<D;d++){float df=a[d]-b[d];ab2+=df*df;a2+=a[d]*a[d];b2+=b[d]*b[d];}
    float den=(1-c*a2)*(1-c*b2);if(den<1e-9f)den=1e-9f;
    return acoshf(1+2*c*ab2/den)/sqrtf(c);
}

int main(void){
    printf("=== k-NN Graph Construction Tests ===\n\n");
    const int N=10,D=8,K=3;
    float c=1.0f;

    float pts[N*D];
    unsigned rs=42u;
    for(int i=0;i<N*D;i++){
        rs=rs*1103515245u+12345u;
        pts[i]=(float)((rs>>16)%2000)/20000.0f-0.05f;
    }
    for(int i=0;i<N;i++){
        float n2=0;for(int d=0;d<D;d++)n2+=pts[i*D+d]*pts[i*D+d];
        if(n2>0.8f){float s=sqrtf(0.8f/n2);for(int d=0;d<D;d++)pts[i*D+d]*=s;}
    }

    printf("  g1_k_neighbors_each...");
    {
        int* idx=NULL,*ptr=NULL,nnz=0;
        CHECK(wubu_knng_build(pts,N,D,K,c,0,&idx,&ptr,&nnz)==0);
        for(int i=0;i<N;i++)
            CHECK(ptr[i+1]-ptr[i]==K);
        free(idx);free(ptr);
    }
    printf("PASS\n");passed++;

    printf("  g2_mutual_symmetric...");
    {
        int* idx=NULL,*ptr=NULL,nnz=0;
        CHECK(wubu_knng_build(pts,N,D,K,c,1,&idx,&ptr,&nnz)==0);
        /* check symmetry via adjacency matrix */
        uint8_t adj[N*N];memset(adj,0,sizeof(adj));
        for(int i=0;i<N;i++)
            for(int p=ptr[i];p<ptr[i+1];p++)
                adj[i*N+idx[p]]=1;
        for(int i=0;i<N;i++)
            for(int j=i+1;j<N;j++)
                CHECK(adj[i*N+j]==adj[j*N+i]);
        free(idx);free(ptr);
    }
    printf("PASS\n");passed++;

    printf("  g3_nearest_correct...");
    {
        int* idx=NULL,*ptr=NULL,nnz=0;
        CHECK(wubu_knng_build(pts,N,D,K,c,0,&idx,&ptr,&nnz)==0);
        /* brute-force verify node 0's nearest */
        float bd=1e30f;int bi=-1;
        for(int j=1;j<N;j++){
            float d=kg_d(pts,pts+j*D,D,c);
            if(d<bd){bd=d;bi=j;}
        }
        int found=0;
        for(int p=ptr[0];p<ptr[0]+K;p++)if(idx[p]==bi)found=1;
        CHECK(found);
        free(idx);free(ptr);
    }
    printf("PASS\n");passed++;

    printf("\n=== Results: %d passed, %d failed ===\n",passed,failed);
    return failed>0?1:0;
}

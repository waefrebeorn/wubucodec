/* GROUP 19: Hyperbolic Graph Neural Network */
#ifndef WUBU_HGNN_H
#define WUBU_HGNN_H
#ifdef __cplusplus
extern "C" {
#endif
double wubu_hgnn_lorentz_dot(const double* a,const double* b,int dim);
double wubu_hgnn_distance(const double* x,const double* y,int dim);
void   wubu_hgnn_project(double* p,int dim,double c);
void   wubu_hgnn_exp_origin(const double* v,double* out,int dim,double c);
void   wubu_hgnn_log_origin(const double* p,double* out,int dim,double c);
int    wubu_hgnn_centroid(const double* points,int n_points,int dim,
                            double c,double* centroid);
int    wubu_hgnn_layer(const double* input,const int* adj,
                          int n_nodes,int n_edges,int feature_dim,
                          const double* weight,double bias,
                          double c,double* output);
#ifdef __cplusplus
}
#endif
#endif

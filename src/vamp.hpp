#pragma once
#include <string>
#include <vector> 
#include <tuple>
#include "data.hpp"
#include "options.hpp"

class vamp{

private:
    int N, M, Mt, C, max_iter, rank, nranks;
    double gam1, gam2, gam_before, eta1, eta2, gam1_add_info;  // linear model precisions
    std::vector<double> gam1s, gam2s, R2trains;
    double tau1, tau2;                          // probit model precisions
    double alpha1, alpha2;                      // Onsager corrections
    double rho;                          // damping factor
    double gamw;                                // linear model noise precision 
    double a_scale;

    std::vector<double> x1_hat, x2_hat, true_signal;
    std::vector<double> z1_hat, z2_hat;
    std::vector<double> y;                              // phenotype vector
    std::vector<double> z1;                             // z1 = A * x1_hat 
    std::vector<double> r1, r2, r2_prev, r1_add_info;
    std::vector<double> p1, p2;
    std::vector<double> cov_eff;                        // covariates in a probit model
    std::vector<double> mu_CG_last;                     // last LMMSE estimate
    

    std::vector<double> probs, probs_before;
    std::vector<double> vars, vars_before;

    double gamma_min = 1e-11;
    double gamma_max = 1e11;
    double probit_var; // hardcoded
    int EM_max_iter; // = 1e5;
    double EM_err_thr; // = 1e-4;
    int CG_max_iter; // = 10;
    int auto_var_max_iter  = 5;// = 50;
    int calc_state_evo = 0;
    int learn_vars;
    int init_est;
    long unsigned int seed;
    unsigned int a_scale_start_iter;
    double damp_max = 1;
    double damp_min = 0.05;
    double stop_criteria_thr; // = 1e-5;
    double gamma_damp;
    double gamw_damp;
    unsigned int sublinear_var; // 0 = standard gam1; 1 = sublinear-sparsity gam1 (see infere())

    std::string model;
    std::string out_dir;
    std::string out_name;
    std::string scheduler;
    std::vector<double> bern_vec;
    std::vector<double> invQ_bern_vec;

    int store_pvals = 1;                                // if .bim file is present we also perform LOCO p-value estimation
    double total_comp_time=0;
    int reverse = 1;
    int use_lmmse_damp = 0;
    int use_freeze=0;

    // cross-validation parameters
    int SBglob, LBglob, redglob;
    int use_cross_val = 0, SB_cross;

    double deltaH;

    // restart variables
    double gam1_init;
    double gamw_init;
    std::string r1_init_file;

    std::string estimate_file;
    std::string freeze_index_file;
    std::string r1_add_info_file;

    int use_tl_lmmse = 0;

    // Multi-source TL inputs
    std::vector<double> gamma_tls;              // size = number of sources
    std::vector<std::string> r_tl_files;        // size = number of sources
    std::vector< std::vector<double> > r_tls;   // r_tls[s][i]

    // Multi-source MAF inputs
    std::string maf_pop1_file;                    // one target MAF file
    std::vector<std::string> maf_pop2_files;      // one source MAF file per TL source

    std::vector<double> maf_pop1_target;          // target MAF values, size M
    std::vector< std::vector<double> > maf_pop2_sources; // source MAF values, [source][marker]

    // Per-marker total TL precision and RHS contribution
    std::vector<double> gamma_tl_vec;   // gamma_tl_vec[i] = sum_s gamma_s_i
    std::vector<double> tl_rhs_vec;     // tl_rhs_vec[i] = sum_s gamma_s_i * r_tls[s][i]

    unsigned int use_maf_tl;   // from --use-maf-tl (default 0)
    double gamma_hyper;        // from --gamma-hyper (default 200)

    // Per-source availability on the joint (union) panel: 1.0 where that source ancestry
    // actually has genotype data for the SNP, 0.0 where the union panel carries it but the
    // ancestry does not. Used to stop absent sources contributing a spurious confident zero.
    std::vector<double> avail_pop1_target;
    std::vector< std::vector<double> > avail_pop2_sources;
    unsigned int tl_mask_missing;

public:

    //******************
    //  CONSTRUCTORS 
    //******************
    vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars,  std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name, std::string model, Options opt = Options());
    vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt);


    //**********************
    // INFERENCE PROCEDURES
    //**********************
    std::vector<double> infere(data* dataset);
    std::vector<double> infere_linear(data* dataset);

    //std::vector<double> predict(std::vector<double> est, data* dataset);


    //********************************************
    // DENOISING PROCEDURES & ONSAGER CALCULATION
    //********************************************
    double g1(double x, double gam1);
    double g1_transfer(double r1, double gam1, double r1_add, double gam1_add, double a_scale);
    double g1d(double x, double gam1);
    double g1d_transfer(double r1, double gam1, double r1_add, double gam1_add, double a_scale);
    double g2d_onsager(double gam2, double tau, data* dataset);


    //************************
    // HYPERPARAMETERS UPDATE
    //************************
    void updatePrior(int verbose);
    void updateNoisePrec(data* dataset);

    std::vector<double> lmmse_mult(std::vector<double> v, double tau, data* dataset, int red = 0);

    std::vector<double> precondCG_solver(std::vector<double> v, double tau, int denoiser, data* dataset, int red = 0);
    std::vector<double> precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int denoiser, data* dataset, int red = 0);    

    void err_measures(data * dataset, int ind);

    void set_SBglob(int SB) { SBglob = SB; }
    void set_LBglob(int LB) { LBglob = LB; }
    void set_gam2 (double gam) { gam2 = gam; }
    std::vector<double> get_cov_eff() const {return cov_eff;}

    /* ======== TL-LMMSE helper functions ======================== */
    std::vector<double> precondCG_TL(const std::vector<double>& b,
                                     double                     gamw,
                                     double                     gam2_eff,
                                     const std::vector<double>& mu0,
                                     data*                      dataset,
                                     int                        red);

    double g2d_onsager_TL(double gam2_orig,
                          double gam2_eff,
                          double gamw,
                          data*  dataset);
    
   };

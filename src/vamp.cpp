#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric> // contains std::accumulate
#include <random>
#include <omp.h>
#include <fstream>
#include <cfloat>
#include "na_lut.hpp"
#include "dotp_lut.hpp"
#include "vamp.hpp"
#include "data.hpp"
#include "vamp_probit.cpp"
#include "vamp_Huber.cpp"
#include "denoiserXXT.cpp"
#include "utilities.hpp"
#include <boost/math/distributions/students_t.hpp> // contains Student's t distribution needed for pvals calculation
#include <stdexcept>


//******************
//  CONSTRUCTORS 
//******************


// -> DESCRIPTION:
//
//      constructor in which all parameters are manually specifies
//
vamp::vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars, std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name, std::string model, Options opt) :
    N(N),
    M(M),
    Mt(Mt),
    C(opt.get_C()),
    gam1(gam1),
    gamw(gamw),
    gam2(0),
    eta1(0),
    eta2(0),
    init_est(opt.get_init_est()),
    max_iter(max_iter),
    rho(rho),
    vars(vars),
    seed(opt.get_seed()),
    probs(probs),
    out_dir(out_dir),
    out_name(out_name),
    true_signal(true_signal),
    estimate_file(opt.get_estimate_file()),
    learn_vars(opt.get_learn_vars()),
    model(model),
    gamma_damp(opt.get_gamma_damp()),
    use_freeze(opt.get_use_freeze()),
    freeze_index_file(opt.get_freeze_index_file()),
    redglob(opt.get_redglob()),
    rank(rank),
    r1_add_info_file(opt.get_r1_add_info_file()),
    scheduler(opt.get_scheduler()),
    gam1_add_info(opt.get_gam1_add_info()),
    a_scale_start_iter(opt.get_a_scale_start_iter()),
    a_scale(opt.get_a_scale()) {
    x1_hat = std::vector<double> (M, 0.0);
    x2_hat = std::vector<double> (M, 0.0);
    r1 = std::vector<double> (M, 0.0);
    r2 = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);
    
    EM_max_iter = opt.get_EM_max_iter();
    EM_err_thr = opt.get_EM_err_thr();
    CG_max_iter = opt.get_CG_max_iter();
    reverse = opt.get_use_XXT_denoiser();
    use_lmmse_damp = opt.get_use_lmmse_damp();
    stop_criteria_thr = opt.get_stop_criteria_thr();
    probit_var = opt.get_probit_var();

    // restart variables
    gam1_init = opt.get_gam1_init();
    gamw_init = opt.get_gamw_init();
    r1_init_file = opt.get_estimate_file();  

    initialize_prior(this->probs, this->vars, N, Mt, rank);

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

}


// -> DESCRIPTION:
//
//      constructor in which all parameters are passed from an Options class
//
vamp::vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt):
    M(M),
    C(opt.get_C()),
    gam1(gam1),
    gamw(gamw),
    gam2(0),
    eta1(0),
    eta2(0),
    rho(opt.get_rho()),
    probs(opt.get_probs()),
    out_dir(opt.get_out_dir()),
    out_name(opt.get_out_name()),
    learn_vars(opt.get_learn_vars()),
    seed(opt.get_seed()),
    true_signal(true_signal),
    model(opt.get_model()),
    redglob(opt.get_redglob()),
    init_est(opt.get_init_est()),
    use_freeze(opt.get_use_freeze()),
    freeze_index_file(opt.get_freeze_index_file()),
    estimate_file(opt.get_estimate_file()),
    store_pvals(opt.get_store_pvals()),
    gamma_damp(opt.get_gamma_damp()),
    rank(rank),
    reverse(opt.get_use_XXT_denoiser()),
    use_lmmse_damp(opt.get_use_lmmse_damp()),
    scheduler(opt.get_scheduler()),
    r1_add_info_file(opt.get_r1_add_info_file()),
    gam1_add_info(opt.get_gam1_add_info()),
    a_scale_start_iter(opt.get_a_scale_start_iter()),
    a_scale(opt.get_a_scale())  {
    N = opt.get_N();
    Mt = opt.get_Mt();
    max_iter = opt.get_iterations();
    x1_hat = std::vector<double> (M, 0.0);
    x2_hat = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);
    r1 = std::vector<double> (M, 0.0);
    r2 = std::vector<double> (M, 0.0);
    EM_max_iter = opt.get_EM_max_iter();
    EM_err_thr = opt.get_EM_err_thr();
    CG_max_iter = opt.get_CG_max_iter();
    stop_criteria_thr = opt.get_stop_criteria_thr();
    vars = opt.get_vars();
    // we scale the signal prior with N since X -> X / sqrt(N)
    probit_var = opt.get_probit_var();

    // restart variables
    gam1_init = opt.get_gam1_init();
    gamw_init = opt.get_gamw_init();
    r1_init_file = opt.get_estimate_file(); 

    initialize_prior(this->probs, this->vars, N, Mt, rank);

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);   
}


//*********************************
// VAMP - MAIN INFERENCE PROCEDURE
//*********************************
std::vector<double> vamp::infere( data* dataset ){

    y = (*dataset).get_phen();

    // we scale mixture variances of effects by N (since the design matrix is scaled by 1/sqrt(N))
    for (int i=0; i<vars.size(); i++)
        vars[i] *= N;

    // forming cross-validation subset, if needed
    if (use_cross_val == 1){
        SB_cross = (int) (0.98 * (*dataset).get_mbytes());
        if (rank == 0){
            std::cout << "N_cross = " << N - 4*SB_cross << std::endl;
            std::cout << "N - N_cross = " << 4*SB_cross << std::endl;
        }
        for (int i=0; i<vars.size(); i++)
            vars[i] *= (double) (4*SB_cross) / (double) N;
    }

    // using different form of LMMSE denoiser, if needed
    if (reverse == 1)
        (*dataset).compute_people_statistics();
    
    // deciding between linear regression and probit regression
    if (!strcmp(model.c_str(), "linear"))
        return infere_linear(dataset);
    else if (!strcmp(model.c_str(), "bin_class"))
        return infere_bin_class(dataset);
    else if (!strcmp(model.c_str(), "robust"))
        return infere_robust(dataset);
    else
        throw "invalid model specification!";

    return std::vector<double> (M, 0.0);
}



//**************
// VAMP LINEAR 
//**************
std::vector<double> vamp::infere_linear(data* dataset){


    // cross-validation variables
    int Nold;
    double prev_R2_cross = -1;
    double rho_cross;
    if (use_cross_val == 1){
        SBglob = 0;
        LBglob = SB_cross;
        redglob = 1;
        Nold = N;
        N = 4*SB_cross;
    }

    // parsing scheduler
    double sch_begin, sch_step_size, sch_first, sch_second;
    int sch_begin_it, sch_num_steps;
    std::string scheduler_mode;
    if (scheduler != "")
    {
        std::vector<std::string> scheduler_parts;
        std::stringstream ss(scheduler);
        std::string item;

        // Split by "::"
        while (std::getline(ss, item, ':')) {
            if (!item.empty() && item.back() == ':') item.pop_back(); // handle "::"
            scheduler_parts.push_back(item);
        }

        if (scheduler_parts.size() < 2) {
            std::cerr << "Invalid input format.\n";
        }

        scheduler_mode  = scheduler_parts[0];

        if (scheduler_mode == "linear" && scheduler_parts.size() == 4) {
            sch_begin = std::stod(scheduler_parts[1]);
            sch_num_steps = std::stoi(scheduler_parts[2]);
            sch_step_size = std::stod(scheduler_parts[3]);

            std::cout << "Mode: Linear\n";
            std::cout << "Begin: " << sch_begin << "\n";
            std::cout << "Num Steps: " << sch_num_steps << "\n";
            std::cout << "Step Size: " << sch_step_size << "\n";
        }
        else if (scheduler_mode == "two-step" && scheduler_parts.size() == 4) {
            sch_first = std::stod(scheduler_parts[1]);
            sch_second = std::stod(scheduler_parts[2]);
            sch_begin_it = std::stod(scheduler_parts[3]);

            std::cout << "Mode: Two-step\n";
            std::cout << "First Scale: " << sch_first << "\n";
            std::cout << "Second Scale: " << sch_second << "\n";
            std::cout << "Begin it: " << sch_begin_it << "\n";
        }
        else {
            std::cerr << "Invalid input format or parameters.\n";
        }
    }

    // loading freeze index file
    std::vector<double> freeze_ind;
    if (use_freeze == 1)
        freeze_ind = read_vec_from_file(freeze_index_file, M, (*dataset).get_S());

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> x1_hat_stored(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);
    alpha1 = 0;

    // filtering a phenotype for nans
    std::vector<double> y =  (*dataset).filter_pheno();

    // Gaussian noise start
    r1 = std::vector<double> (M, 0.0);

    // restart option
    if (gam1_init != -1){
        gam1 = gam1_init;
        gamw = gamw_init;
        std::vector<double> r1_init = mpi_read_vec_from_file(r1_init_file, M, (*dataset).get_S()); // file name ends in .bin
        for (int i=0; i<M; i++)
            r1_init[i] /= sqrt(N);
        r1 = r1_init;   
    }

    // in case we initialize with an estimate of the signal
    if (init_est == 1){
        std::vector<double> x_est;
        int pos_dot = estimate_file.find(".");
        std::string end_est_file_name = estimate_file.substr(pos_dot + 1);
        if (end_est_file_name == "bin")
            x_est = mpi_read_vec_from_file(estimate_file, M, (*dataset).get_S());
        else
            x_est = read_vec_from_file(estimate_file, M, (*dataset).get_S());

        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N );

        x1_hat = x_est;
        r1 = x_est;
    }

    if (r1_add_info_file != ""){
        int pos_dot = r1_add_info_file.find(".");
        std::string end_r1_add_file_name = r1_add_info_file.substr(pos_dot + 1);
        if (end_r1_add_file_name == "bin")
            r1_add_info = mpi_read_vec_from_file(r1_add_info_file, M, (*dataset).get_S());
        else
            r1_add_info = read_vec_from_file(r1_add_info_file, M, (*dataset).get_S());
        for (int i = 0; i < M; i++)
            r1_add_info[i] *= sqrt( (double) N );
        gam1_add_info /= (double) N;

        if (rank == 0){
            std::cout << "gam1_add_info = " << gam1_add_info << "\n";
            std::cout << "r1_add_info[0] = " << r1_add_info[0] << "\n";
            std::cout << "a_scale = " << a_scale << "\n";
        }
    }

    // starting VAMP iterations
    for (int it = 1; it <= max_iter; it++)
    {    

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // %%%%%%%%%%% Denoising step %%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double start_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;

        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;

        // keeping value of Onsager from a previous iteration
        double alpha1_prev = alpha1;

        // re-estimating the error variance
        double gam1_reEst_prev;
        int it_revar = 1;

        for (; it_revar <= auto_var_max_iter; it_revar++){

            // new signal estimate
            // if (rank == 0)
                // std::cout << "rank = " << rank <<  ", before denoiser \n";
            for (int i = 0; i < M; i++){
                if (r1_add_info_file != "" && a_scale_start_iter <= it)
                    x1_hat[i] = g1_transfer(r1[i], gam1, r1_add_info[i], gam1_add_info, a_scale);
                else
                    x1_hat[i] = g1(r1[i], gam1);
                // break;
            }
            // if (rank == 0)
            //     std::cout << "rank = 0, after denoiser, x1_hat[0] = " << x1_hat[0] << ", x1_hat[11] = " << x1_hat[11] << "\n";

            if (it==1 && init_est==1)
                x1_hat = r1;

            std::vector<double> x1_hat_m_r1 = x1_hat;
            //std::transform (x1_hat_m_r1.begin(), x1_hat_m_r1.end(), r1.begin(), x1_hat_m_r1.begin(), std::minus<double>());
            for (int i0 = 0; i0 < x1_hat_m_r1.size(); i0++)
                x1_hat_m_r1[i0] = x1_hat_m_r1[i0] - r1[i0];

            // new MMSE estimate
            double sum_d = 0;
            for (int i=0; i<M; i++)
            {
                if (r1_add_info_file != "")
                    x1_hat_d[i] = g1d_transfer(r1[i], gam1, r1_add_info[i], gam1_add_info, a_scale);
                else 
                    x1_hat_d[i] = g1d(r1[i], gam1);
                if (!use_freeze || (use_freeze && freeze_ind[i]==0))
                    sum_d += x1_hat_d[i];
            }

            alpha1 = 0;
            MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            alpha1 /= Mt;
            // if (rank == 0)
            //     std::cout << "alpha1 = " << alpha1 << "\n";
            eta1 = gam1 / alpha1;

            if (it <= 1)
                break;


            gam1_reEst_prev = gam1;
            if (it > 1)
                gam1 = std::min( std::max(  1.0 / (1.0/eta1 + l2_norm2(x1_hat_m_r1, 1)/Mt), gamma_min ), gamma_max );
            else
                break;

            updatePrior(0);

            if (rank == 0 && it_revar % 1 == 0)
                std::cout << "[old] it_revar = " << it_revar << ": gam1 = " << gam1 << std::endl;

            if ( abs(gam1 - gam1_reEst_prev) < 1e-3 )
                break;
            
        }

        // saving gam1 estimates
        gam1s.push_back(gam1);

        if (rank == 0)
            std::cout << "A total of " << std::max(it_revar - 1,1) << " variance and prior tuning iterations were performed" << std::endl;
        

        // damping on the level of x1
        if (it > 1){ 

            std::vector<double> x1_hat_temp = x1_hat;
            // change in damping
            for (int i = 0; i < M; i++)
                if (!use_freeze || (use_freeze && freeze_ind[i] == 0))
                    x1_hat[i] = rho * x1_hat[i] + (1-rho) * x1_hat_prev[i];

            //for (int i = 0; i < M; i++)
            //    x1_hat[i] = rho * x1_hat[i] + (1-rho) * x2_hat[i];

            if (use_cross_val == 1){
                int cross_it = 1;
                double currR2;
                rho_cross = rho;
                
                while(cross_it <= 25){

                    std::vector<double> z_cross =  (*dataset).Ax(x1_hat.data(), SB_cross, (*dataset).get_mbytes() - SB_cross);
                    for (int j=0; j<z_cross.size(); j++)
                        z_cross[j] *= sqrt( (double) ( 4 * ((*dataset).get_mbytes() - SB_cross) ) / (double) (4*SB_cross) );

                    double l2_pred_err2 = 0;

                    for (int i0 = SB_cross; i0 < (*dataset).get_mbytes(); i0++)
                        for (int k=0; k<4; k++)
                            if (4*i0 + k < Nold)
                                l2_pred_err2 += (y[4*i0 + k] - z_cross[4*(i0-SB_cross) + k]) * (y[4*i0 + k] - z_cross[4*(i0-SB_cross) + k]);

                    std::vector<double> y_cross (Nold - 4*SB_cross, 0.0); 
                    for (int i=0; i<y_cross.size(); i++)
                        y_cross[i] = y[4*SB_cross + i];

                    double stdev = calc_stdev(y_cross);
                    currR2 = 1 - l2_pred_err2 / ( stdev * stdev * y_cross.size() );
                    if (rank == 0)
                        std::cout << "[cross_val, it = " << cross_it << "] R2 = " << currR2 << std::endl;
                    
                    if (it == 1){
                        prev_R2_cross = currR2;
                        break;
                    }

                    if (currR2 < 1 * prev_R2_cross){
                        rho_cross *= 0.90;
                        if (rank == 0)
                            std::cout << "[cross_val] rho_cross = " << rho_cross << std::endl;
                        for (int i = 0; i < M; i++)
                            x1_hat[i] = rho_cross * x1_hat_temp[i] + (1-rho_cross) * x1_hat_prev[i];
                    }
                    else{
                        prev_R2_cross = currR2;
                        break;
                    }   

                    cross_it++;
                }

                //if (currR2 < prev_R2_cross)
                //    break;

            }
            
            if (use_cross_val == 1 && it > 1)
                alpha1 = rho_cross * alpha1 + (1-rho_cross) * alpha1_prev;
            else if (it > 1)
                alpha1 = rho * alpha1 + (1-rho) * alpha1_prev;
                //alpha1 = rho * alpha1 + (1-rho) * alpha2;


            // damping on the level of prior parameters
            //for (int i = 0; i < probs.size(); i++){
            //    probs[i] = rho * probs[i] + (1-rho) * probs_before[i];
            //    vars[i] = rho * vars[i] + (1-rho) * vars_before[i];
            //}
        }

        double start_z1= MPI_Wtime();
        if (use_cross_val == 1)
            z1 = (*dataset).Ax(x1_hat.data(), SBglob, LBglob);
        else
            z1 = (*dataset).Ax(x1_hat.data());
        double end_z1= MPI_Wtime();

        if (rank == 0)
            std::cout << "time needed to calculate z1 = " << end_z1 - start_z1 << " seconds" <<  std::endl;

        std::string filepath_out_z1 = out_dir + out_name + "_z1_it_" + std::to_string(it) + ".csv";
        store_vec_to_file(filepath_out_z1, z1);

        if (rank == 0)
            std::cout << "filepath_out_z1 = " << filepath_out_z1 <<  std::endl;

        if (rank == 0)
           std::cout << "rho = " << rho << std::endl;

        

        // saving x1_hat
        double scale = sqrt(N);
        double start_saving = MPI_Wtime();
        std::string filepath_out = out_dir + out_name + "_it_" + std::to_string(it) + ".bin";
        int S = (*dataset).get_S();
        for (int i0=0; i0<x1_hat_stored.size(); i0++)
            x1_hat_stored[i0] =  x1_hat[i0] / scale;
        mpi_store_vec_to_file(filepath_out, x1_hat_stored, S, M);

        if (rank == 0)
           std::cout << "x1_hat filepath_out is " << filepath_out << std::endl;

        std::string filepath_out_r1 = out_dir + out_name + "_r1_it_" + std::to_string(it) + ".bin";
        std::vector<double> r1_stored = r1;
        for (int i0=0; i0<r1_stored.size(); i0++)
            r1_stored[i0] =  r1[i0] / scale;
        mpi_store_vec_to_file(filepath_out_r1, r1_stored, S, M);

        if (rank == 0)
           std::cout << "r1 filepath_out is " << filepath_out_r1 << std::endl;

        double end_saving = MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to save beta1 to an external file = " << end_saving - start_saving << " seconds" <<  std::endl;

        gam_before = gam2;
        gam2 = std::min(std::max(eta1 - gam1, gamma_min), gamma_max);

        if (rank == 0){
            std::cout << "eta1 = " << eta1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
        }

        if (use_lmmse_damp == 1)
            r2_prev = r2;

        // onsager approx
        r2_prev = r2;

        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;

        if (use_lmmse_damp == 1){
            //double xi = std::min(2 * std::min(alpha1, alpha2), 1.0);
            //if (rank == 0)
            //    std::cout << "xi = " << xi << std::endl;
            double xi = std::min(2*rho, 1.0);
            if (it > 1){
                //for (int i = 0; i < M; i++)
                //    r2[i] = xi * r2[i] + (1-xi) * r2_prev[i];
                gam2 = 1.0 / pow( xi / sqrt(gam2) + (1-xi) / sqrt(gam_before), 2);
            } 
        }

        // we try with larger rhos if damping criteria allows it
        double xi = std::min(2 * std::min(alpha1, alpha2), 1.0);
        rho = std::max(rho, xi);

        // if the true value of the signal is known, we print out the true gam2
        double se_dev = 0;
        for (int i0=0; i0<M; i0++){
            se_dev += (r2[i0] - scale*true_signal[i0])*(r2[i0] - scale*true_signal[i0]);
        }
        double se_dev_total = 0;
        MPI_Allreduce(&se_dev, &se_dev_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "true gam2 = " << Mt / se_dev_total << std::endl;

        
        double start_prior_up = MPI_Wtime();

        // new place for prior update
        if (auto_var_max_iter == 0 || it <=1)
            updatePrior(1);

        double end_prior_up = MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to calculate conditional expectation = " << end_prior_up - start_prior_up << " seconds" <<  std::endl;
    
        err_measures(dataset, 1);

        double end_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << "denoising step took " << end_denoising - start_denoising << " seconds." << std::endl;


        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // %%%%%%%%%% LMMSE step %%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        // saving r2
        std::string filepath_out_r2 = out_dir + out_name + "_r2_it_" + std::to_string(it) + ".bin";
        std::vector<double> r2_stored = r2;
        for (int i0=0; i0<r2_stored.size(); i0++)
            r2_stored[i0] =  r2[i0] / scale;
        mpi_store_vec_to_file(filepath_out_r2, r2_stored, S, M);

        if (rank == 0)
           std::cout << "r2 filepath_out is " << filepath_out_r2 << std::endl;

        double start_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "______________________" << std::endl<< "->LMMSE" << std::endl;


        // gamma_damp
        // gam2 = gam2 * gamma_damp;

        // running conjugate gradient solver to compute LMMSE
        double start_CG = MPI_Wtime();
        if (reverse == 0){

            std::vector<double> v;
            if (use_cross_val == 1)
                v = (*dataset).ATx(y.data(), SBglob, LBglob);
            else
                v = (*dataset).ATx(y.data());

            for (int i = 0; i < M; i++)
                v[i] = gamw * v[i] + gam2 * r2[i];

            if (it == 1)
                x2_hat = precondCG_solver(v, std::vector<double>(M, 0.0), gamw, 1, dataset, redglob); // precond_change!
            else
                x2_hat = precondCG_solver(v, mu_CG_last, gamw, 1, dataset, redglob); // precond_change!

        } 
        else if (reverse == 1){

            if (it == 1)
                mu_CG_last = std::vector<double> (4*(*dataset).get_mbytes(), 0.0);

            x2_hat = lmmse_denoiserAAT(r2, mu_CG_last, dataset);

        }

        std::string filepath_out_x2 = out_dir + out_name + "_it_" + std::to_string(it) + "_x2_hat.bin";
        std::vector<double> x2_hat_stored = x2_hat;
        for (int i0=0; i0<x2_hat_stored.size(); i0++)
            x2_hat_stored[i0] =  x2_hat[i0] / scale;
        mpi_store_vec_to_file(filepath_out_x2, x2_hat_stored, S, M);
        if (rank == 0)
           std::cout << "x2_hat filepath_out is " << filepath_out_x2 << std::endl;

        
        double end_CG = MPI_Wtime();

        if (rank == 0)
            std::cout << "CG took "  << end_CG - start_CG << " seconds." << std::endl;


        double start_onsager = MPI_Wtime();
        alpha2 = g2d_onsager(gam2, gamw, dataset);
        double end_onsager = MPI_Wtime();

        if (rank == 0)
            std::cout << "onsager took "  << end_onsager - start_onsager << " seconds." << std::endl;
        
        if (rank == 0)
            std::cout << "alpha2 = " << alpha2 << std::endl;

        // gamma_damp back
        // gam2 = gam2 / gamma_damp;

        // onsager approx
        if (it > 1){
            std::vector<double> r2_m_r2prev = r2;
            for (int i=0; i<M; i++)
                r2_m_r2prev[i] -= r2_prev[i];

            double onsager_approx = inner_prod(x2_hat, r2_m_r2prev, 1) / inner_prod(r2, r2_m_r2prev, 1);
            if (rank == 0)
                std::cout << "onsager approx = " << onsager_approx << std::endl;

            // polynomial onsager approximator
            std::vector<double> Xr2 = (*dataset).Ax(r2.data());
            std::vector<double> r2_m_x2hat = r2;
            for (int i=0; i<M; i++)
                r2_m_x2hat[i] -= x2_hat[i];
            std::vector<double> Xx2hat_m_y = (*dataset).Ax(x2_hat.data());
            for (int i=0; i<N; i++)
                Xx2hat_m_y[i] -= y[i];

            double u1 = (l2_norm2(r2_m_x2hat,1) - 1.0/gam2 - l2_norm2(Xx2hat_m_y,0) + (double) N / (double) Mt / gamw) / Mt;
            double u2 = (inner_prod(r2,r2,1) - inner_prod(x2_hat,r2,1) - inner_prod(y, Xr2, 0) + inner_prod((*dataset).Ax(x2_hat.data()), Xr2,0) )/Mt*2;
            double u3 = l2_norm2(r2,1)/Mt - l2_norm2(Xr2,0)/Mt;
            // alpha2 = onsager_approx;
            if (rank == 0){
                std::cout << "u1 = " << u1 << ", u2 = " << u2 << ", u3 = " << u3 << std::endl;
                double polyest1 = (-u2 + sqrt(u2*u2 - 4*u1*u3))/2/u3;
                double polyest2 = (-u2 - sqrt(u2*u2 - 4*u1*u3))/2/u3;
                std::cout << "polyest1 = " << polyest1 << ", polyest2 = " << polyest2 << std::endl;
                double polyest;
                if (u3<0)
                    polyest = std::max(polyest1,polyest2);
                else
                    polyest = std::min(polyest1,polyest2);
                std::cout << "extremum = " << -u2/2/u3 << std::endl;
                std::cout << "polyest = " << polyest << std::endl;
            }
        }
        eta2 = gam2 / alpha2;


        // re-estimating gam2 <- new
        std::vector<double> x2_hat_m_r2 = x2_hat;
        for (int i0 = 0; i0 < x2_hat_m_r2.size(); i0++)
            x2_hat_m_r2[i0] = x2_hat_m_r2[i0] - r2[i0];
        // double gam2_before = gam2;

        if (auto_var_max_iter >= 1 && it > 2){
            gam2 = std::min( std::max(  1 / (1/eta2 + l2_norm2(x2_hat_m_r2, 1)/Mt), gamma_min ), gamma_max );
        }

        if (rank == 0)
            std::cout << "gam2 re-est = " << gam2 << std::endl;

        gam2s.push_back(gam2);

        // gam_before = gam1;
        
        gam1 = std::min( std::max( eta2 - gam2, gamma_min ), gamma_max );
        

        //r1_prev = r1;
        for (int i = 0; i < M; i++)
            r1[i] = (eta2 * x2_hat[i] - gam2 * r2[i]) / gam1;
            // r1[i] = rho_it * (eta2 * x2_hat[i] - gam2 * r2[i]) / gam1 + (1-rho_it) * r1_prev[i];

        // gam1 = rho_it * gam1 + (1-rho_it) * gam_before;

        if (rank == 0)
            std::cout << "gam1 = " << gam1 << std::endl;

        // if the true value of the signal is known, we print out the true gam1
        double se_dev1 = 0;
        for (int i0=0; i0<M; i0++)
            se_dev1 += (r1[i0]- scale*true_signal[i0])*(r1[i0]- scale*true_signal[i0]);
        double se_dev_total1 = 0;
        MPI_Allreduce(&se_dev1, &se_dev_total1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "true gam1 = " << Mt / se_dev_total1 << std::endl; 

        // learning a noise precision parameter
        updateNoisePrec(dataset);

        // printing out error measures
        err_measures(dataset, 2);
        
        double end_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "lmmse step took "  << end_lmmse_step - start_lmmse_step << " seconds." << std::endl;
        
        //rho = rho * scale_rho; 

        // stopping criteria
        std::vector<double> x1_hat_diff = x1_hat;
        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat_diff[i0];

        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "VAMP stopping criteria fulfilled with threshold = " << stop_criteria_thr << "." << std::endl;
            break;
        }
        
        if (rank == 0)
            std::cout << "total iteration time = " << end_denoising - start_denoising + end_lmmse_step - start_lmmse_step << std::endl;
        total_comp_time += end_denoising - start_denoising + end_lmmse_step - start_lmmse_step;
        if (rank == 0)
            std::cout << "total computation time so far = " << total_comp_time << std::endl;
        
        if (rank == 0)
            std::cout << std::endl << std::endl;
    }

    if (store_pvals == 1){

        // calculating p-values using LOO method
        std::string filepath_out_pvals = out_dir + out_name + "_pvals.bin";
        std::vector< std::vector<double> > pvals = (*dataset).pvals_calc(std::vector< std::vector<double> > {z1}, y, std::vector< std::vector<double> > {x1_hat}, std::vector< std::string > {filepath_out_pvals});
        if (rank == 0)
            std::cout << "filepath_out_pvals = " << filepath_out_pvals << std::endl;

        // calculating p-values using LOCO method, if .bim file is specified
        if ((*dataset).get_bimfp() != ""){
            std::string filepath_out_pvals_LOCO = out_dir + out_name;
            std::vector< std::vector<double> > pvals_LOCO = (*dataset).pvals_calc_LOCO(std::vector< std::vector<double> > {z1}, y, std::vector< std::vector<double> > {x1_hat}, std::vector< std::string > {filepath_out_pvals_LOCO});
            if (rank == 0)
                std::cout << "filepath_out_pvals_LOCO = " << filepath_out_pvals_LOCO << std::endl;
        }
    }
    
    // saving gam1s
    std::string filepath_out_gam1s = out_dir + out_name + "_gam1s.csv";
    store_vec_to_file(filepath_out_gam1s, gam1s);
    if (rank == 0)
        std::cout << "gam1s filepath_out is " << filepath_out_gam1s << std::endl;

    // saving gam2s
    std::string filepath_out_gam2s = out_dir + out_name + "_gam2s.csv";
    store_vec_to_file(filepath_out_gam2s, gam2s);
    if (rank == 0)
        std::cout << "gam2s filepath_out is " << filepath_out_gam2s << std::endl;

    // saving R2 train
    std::string filepath_out_R2trains = out_dir + out_name + "_R2trains.csv";
    store_vec_to_file(filepath_out_R2trains, R2trains);
    if (rank == 0)
        std::cout << "R2trains filepath_out is " << filepath_out_R2trains << std::endl;

    int maxR2trainsIndex = std::max_element(R2trains.begin(),R2trains.end()) - R2trains.begin();
    if (rank == 0)
        std::cout << "index of max train R2 iteration = " << maxR2trainsIndex << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    // returning scaled version of the effects
    return x1_hat_stored;          
}

double vamp::g1_transfer(double r1, 
                     double gam1, 
                     double r1_add,
                     double gam1_add,
                     double a_scale) {

    double lambda = 1 - probs[0];

    std::vector<double> variances(vars.begin() + 1, vars.end());
    std::vector<double> omegas(probs.begin() + 1, probs.end());
    for (int j = 0; j < omegas.size(); j++) // omegas and variances are of length L
        omegas[j] /= lambda;

    // Compute sigma2_meta = 1.0 / (sum(a_scale * gam1s) + 1.0/vars)
    double sum_a_gam1s = a_scale * gam1 + (1-a_scale) * gam1_add;
    std::vector<double> sigma2_meta(variances.size());
    for (size_t i = 0; i < sigma2_meta.size(); ++i) 
        sigma2_meta[i] = 1.0 / (sum_a_gam1s + 1.0 / variances[i]);

    // Compute mu_meta = np.inner(rs, a * gam1s) * sigma2_meta
    std::vector<double> mu_meta(variances.size());
    for (size_t i = 0; i < variances.size(); ++i) 
        mu_meta[i] = (r1 * gam1 * a_scale + r1_add * gam1_add * (1-a_scale)) * sigma2_meta[i];

    // Find max_ind = argmax(mu_meta^2 / sigma2_meta)
    std::vector<double> ratio(mu_meta.size());
    std::transform(mu_meta.begin(), mu_meta.end(), sigma2_meta.begin(), ratio.begin(),
                   [](double mu, double sigma) { return (mu * mu) / sigma; } );

    int max_ind = std::distance(ratio.begin(), std::max_element(ratio.begin(), ratio.end()));

    // Compute EXP
    std::vector<double> EXP(mu_meta.size());
    double max_mu_sq = mu_meta[max_ind] * mu_meta[max_ind];

    std::transform(mu_meta.begin(), mu_meta.end(), sigma2_meta.begin(), EXP.begin(),
                   [max_mu_sq, max_ind, &sigma2_meta](double mu, double sigma) {
                       return std::exp(0.5 * ( (mu * mu * sigma2_meta[max_ind] - max_mu_sq * sigma) /
                                              (sigma * sigma2_meta[max_ind]) ));
                   });

    double Num = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) {
        Num += omegas[i] * EXP[i] * mu_meta[i] * std::sqrt(sigma2_meta[i] / variances[i]);
    }

    // Multiply by lambda
    Num *= lambda;

    // Compute EXP2
    double EXP2 = std::exp(-0.5 * (mu_meta[max_ind] * mu_meta[max_ind] / sigma2_meta[max_ind]));

    // Compute Den
    double sum_term = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) {
        sum_term += omegas[i] * EXP[i] * std::sqrt(sigma2_meta[i] / variances[i]);
        // std::cout << "std::sqrt(sigma2_meta[i] / vars[i]) = " << std::sqrt(sigma2_meta[i] / vars[i]) << std::endl;
        // std::cout << "EXP[i] = " << EXP[i] << std::endl;
        // std::cout << "omegas[i] = " << omegas[i] << std::endl;
    }
    
    // std::cout << "sum_term = " << sum_term << std::endl;
    // std::cout << "EXP2 = " << EXP2 << std::endl;
    // std::cout << "lambda = " << lambda << std::endl;

    // Compute Den
    double Den = (1 - lambda) * EXP2 + lambda * sum_term;

    return Num / Den;
}

double vamp::g1d_transfer(double r1, 
                          double gam1, 
                          double r1_add,
                          double gam1_add,
                          double a_scale) {

    double lambda = 1 - probs[0];

    std::vector<double> variances(vars.begin() + 1, vars.end());
    std::vector<double> omegas(probs.begin() + 1, probs.end());
    for (int j = 0; j < omegas.size(); j++) // omegas and variances are of length L
        omegas[j] /= lambda;

    // Compute sigma2_meta = 1.0 / (sum(a_scale * gam1s) + 1.0/vars)
    double sum_a_gam1s = a_scale * gam1 + (1-a_scale) * gam1_add;
    std::vector<double> sigma2_meta(variances.size());
    for (size_t i = 0; i < sigma2_meta.size(); ++i) 
        sigma2_meta[i] = 1.0 / (sum_a_gam1s + 1.0 / variances[i]);

    // Compute mu_meta = np.inner(rs, a * gam1s) * sigma2_meta
    std::vector<double> mu_meta(variances.size());
    for (size_t i = 0; i < variances.size(); ++i) 
        mu_meta[i] = (r1 * gam1 * a_scale + r1_add * gam1_add * (1-a_scale)) * sigma2_meta[i];

    // Find max_ind = argmax(mu_meta^2 / sigma2_meta)
    std::vector<double> ratio(mu_meta.size());
    std::transform(mu_meta.begin(), mu_meta.end(), sigma2_meta.begin(), ratio.begin(),
                    [](double mu, double sigma) { return (mu * mu) / sigma; } );

    int max_ind = std::distance(ratio.begin(), std::max_element(ratio.begin(), ratio.end()));

    // Compute EXP
    std::vector<double> EXP(mu_meta.size());
    double max_mu_sq = mu_meta[max_ind] * mu_meta[max_ind];

    std::transform(mu_meta.begin(), mu_meta.end(), sigma2_meta.begin(), EXP.begin(),
                    [max_mu_sq, max_ind, &sigma2_meta](double mu, double sigma) {
                        return std::exp(0.5 * ((mu * mu * sigma2_meta[max_ind] - max_mu_sq * sigma) /
                                                (sigma * sigma2_meta[max_ind])));
                    });

    double Num = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) {
        Num += omegas[i] * EXP[i] * mu_meta[i] * std::sqrt(sigma2_meta[i] / variances[i]);
    }

    // Multiply by lambda
    Num *= lambda;

    // std::cout << "Num = " << Num << "\n";

    // Compute EXP2
    double EXP2 = std::exp(-0.5 * (mu_meta[max_ind] * mu_meta[max_ind] / sigma2_meta[max_ind]));

    // Compute Den
    double sum_term = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) 
        sum_term += omegas[i] * EXP[i] * std::sqrt(sigma2_meta[i] / variances[i]);

    // Compute Den
    double Den = (1 - lambda) * EXP2 + lambda * sum_term;

    // std::cout << "Den = " << Den << "\n";

    // Compute DerNum
    double DerNum = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) {
        DerNum += omegas[i] * EXP[i] * (mu_meta[i] * mu_meta[i] + sigma2_meta[i]) * a_scale * gam1 * std::sqrt(sigma2_meta[i] / variances[i]);
    }
    DerNum *= lambda;

    // std::cout << "DerNum = " << DerNum << "\n";

    // Compute DerDen
    double DerDen = 0.0;
    for (size_t i = 0; i < omegas.size(); ++i) {
        DerDen += omegas[i] * mu_meta[i] * EXP[i] * a_scale * gam1 * std::sqrt(sigma2_meta[i] / variances[i]);
    }
    DerDen *= lambda;

    // std::cout << "DerDen = " << DerDen << "\n";

    // Compute final result
    double result = (DerNum * Den - DerDen * Num) / (Den * Den);

    // std::cout << "result = " << result << "\n";

    return result;

}

double vamp::g1(double y, double gam1) { //fchecked
    
    double sigma = 1 / gam1;

    double eta_max = *(std::max_element(vars.begin(), vars.end()));

    double pk = 0, pkd = 0, val;

    if (sigma < 1e-10 && sigma > -1e-10) {
        return y;
    }

    for (int i = 0; i < probs.size(); i++){

        double expe_sum =  - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );

        double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

        pk = pk + z;

        z = z / ( vars[i] + sigma ) * y;

        pkd = pkd - z; 

    }

    val = (y + sigma * pkd / pk);
    
    return val;
}

double vamp::g1d(double y, double gam1) { 
    
        double sigma = 1 / gam1;

        double eta_max = *std::max_element(vars.begin(), vars.end());

        double pk = 0, pkd = 0, pkdd = 0;
        
        if (sigma < 1e-10 && sigma > -1e-10) {
            return 1;
        }

        for (int i = 0; i < probs.size(); i++){

            double expe_sum = - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );

            double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

            pk = pk + z;

            z = z / ( vars[i] + sigma ) * y;

            pkd = pkd - z;

            double z2 = z / ( vars[i] + sigma ) * y;

            pkdd = pkdd - probs[i] / pow( vars[i] + sigma, 1.5 ) * exp( expe_sum ) + z2;
            
        } 

        double val = (1 + sigma * ( pkdd / pk - pow( pkd / pk, 2) ) );

        return val;
}

double vamp::g2d_onsager(double gam2, double tau, data* dataset) { // shared between linear and binary classification model
    
    // std::random_device rd;

    std::mt19937 rd{seed + (long unsigned int) (*dataset).get_S()}; // rng

    std::bernoulli_distribution bern(0.5);

    bern_vec = std::vector<double> (M, 0.0);

    for (int i = 0; i < M; i++)
        bern_vec[i] = (2*bern(rd) - 1) / sqrt(Mt); // Bernoulli variables are sampled independently

    invQ_bern_vec = precondCG_solver(bern_vec, tau, 0, dataset, redglob); // precond_change

    double onsager = gam2 * inner_prod(bern_vec, invQ_bern_vec, 1); // because we want to calculate gam2 * Tr[(gamw * X^TX + gam2 * I)^(-1)] / Mt

    return onsager;    
}


void vamp::updateNoisePrec(data* dataset){

    //filtering for NAs
    y = (*dataset).filter_pheno();  

    std::vector<double> temp = (*dataset).Ax(x2_hat.data());

    for (int i = 0; i < N; i++)  // because length(y) = N
        temp[i] -= y[i];
    
    double temp_norm2 = l2_norm2(temp, 0); 
    
    std::vector<double> trace_corr_vec_N;
    std::vector<double> trace_corr_vec_M;

    if (redglob == 1){
        trace_corr_vec_N = (*dataset).Ax(invQ_bern_vec.data(), SBglob, LBglob);

        trace_corr_vec_M = (*dataset).ATx(trace_corr_vec_N.data(), SBglob, LBglob);
    }
    else{
        trace_corr_vec_N = (*dataset).Ax(invQ_bern_vec.data());

        trace_corr_vec_M = (*dataset).ATx(trace_corr_vec_N.data());
    }
    
    double trace_corr = inner_prod(bern_vec, trace_corr_vec_M, 1) * Mt; // because we took u ~ Bern({-1,1} / sqrt(Mt), 1/2)

    if (rank == 0){
        std::cout << "l2_norm2(temp) / N = " << temp_norm2 / N << std::endl;
        std::cout << "trace_correction / N = " << trace_corr / N << std::endl;
    }

    gamw = (double) N / (temp_norm2 + trace_corr);

}

void vamp::updatePrior(int verbose = 1) {
    
        double noise_var = 1 / gam1;

        double lambda = 1 - probs[0];

        std::vector<double> omegas = probs;
        for (int j = 1; j < omegas.size(); j++) // omegas is of length L
            omegas[j] /= lambda;
                       
        // calculating normalized beta and pin
        int it;

        for (it = 0; it < EM_max_iter; it++){

            double max_sigma = *std::max_element(vars.begin(), vars.end()); // std::max_element returns iterators, not values

            std::vector<double> probs_prev = probs;
            std::vector<double> vars_prev = vars;
            std::vector< std::vector<double> > gammas;
            std::vector< std::vector<double> > beta;
            std::vector<double> pin(M, 0.0);
            std::vector<double> v;

            for (int i = 0; i < M; i++){

                std::vector<double> temp; // of length (L-1)

                std::vector<double> temp_gammas;

                for (int j = 1; j < probs.size(); j++ ){

                    double num = lambda * omegas[j] * exp( - pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var) ) / sqrt(vars[j] + noise_var) / sqrt(2 * M_PI);
                    
                    double num_gammas = gam1 * r1[i] / ( 1 / vars[j] + gam1 );   

                    temp.push_back(num);

                    temp_gammas.push_back(num_gammas);
                }

                double sum_of_elems = std::accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));
            
                for (int j = 0; j < temp.size(); j++ )
                    temp[j] /= sum_of_elems;
                
                beta.push_back(temp);

                gammas.push_back(temp_gammas);
                
                pin[i] = 1 / ( 1 + (1-lambda) / sqrt(2 * M_PI * noise_var) * exp( - pow(r1[i], 2) / 2 * max_sigma / noise_var / (noise_var + max_sigma) ) / sum_of_elems );

            } 

            for (int j = 1; j < probs.size(); j++)
                v.push_back( 1.0 / ( 1.0 / vars[j] + gam1 ) ); // v is of size (L-1) in the end
            
            lambda = accumulate(pin.begin(), pin.end(), 0.0); // / pin.size();

            double lambda_total = 0;

            MPI_Allreduce(&lambda, &lambda_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            lambda = lambda_total / Mt;
        
            for (int i = 0; i < M; i++){
                for (int j = 0; j < (beta[0]).size(); j++ ){
                    gammas[i][j] = beta[i][j] * ( gammas[i][j] * gammas[i][j] + v[j] );
                }
            }

            //double sum_of_pin = std::accumulate(pin.begin(), pin.end(), decltype(pin)::value_type(0));
            double sum_of_pin = lambda_total;

            for (int j = 0; j < (beta[0]).size(); j++){ // of length (L-1)
                double res = 0, res_gammas = 0;
                for (int i = 0; i < M; i++){
                    res += beta[i][j] * pin[i];
                    res_gammas += gammas[i][j] * pin[i];
                }

                double res_gammas_total = 0;
                double res_total = 0;
                MPI_Allreduce(&res_gammas, &res_gammas_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&res, &res_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                if (learn_vars == 1)
                    vars[j+1] = res_gammas_total / res_total;
                omegas[j+1] = res_total / sum_of_pin;
                probs[j+1] = lambda * omegas[j+1];

            }

            probs[0] = 1 - lambda;
        
            double distance_probs = 0, norm_probs = 0;
            double distance_vars = 0, norm_vars = 0;

            for (int j = 0; j < probs.size(); j++){

                distance_probs += ( probs[j] - probs_prev[j] ) * ( probs[j] - probs_prev[j] );

                norm_probs += probs[j] * probs[j];

                distance_vars += ( vars[j] - vars_prev[j] ) * ( vars[j] - vars_prev[j] );

                norm_vars += vars[j] * vars[j];
            }
            double dist_probs = sqrt(distance_probs / norm_probs);
            double dist_vars = sqrt(distance_vars / norm_vars);

            if (verbose == 1)
                if (rank == 0)
                    std::cout << "it = " << it << ": dist_probs = " << dist_probs << " & dist_vars = " << dist_vars << std::endl;
            if ( dist_probs < EM_err_thr  && dist_vars < EM_err_thr )
                break;   
        }
    
        if (verbose == 1)
            if (rank == 0)  
                std::cout << "Final number of prior EM iterations = " << std::min(it + 1, EM_max_iter) << " / " << EM_max_iter << std::endl;


        // merging close variances

        for (int j = 0; j < vars.size(); j++){
            for (int k = j+1; k < vars.size(); k++){

                double denom;
                if (vars[j] != 0)
                    denom = std::min(vars[j], vars[k]);
                else
                    denom = 1e-7;

                if ( abs(vars[j] - vars[k]) / denom < 5e-1 ){
                    double sum2probs = probs[j] + probs[k];
                    vars.erase(vars.begin() + k);
                    probs.erase(probs.begin() + k);
                    probs[j] = sum2probs;
                    k--;
                }
            }
        }
}

std::vector<double> vamp::lmmse_mult(std::vector<double> v, double tau, data* dataset, int red){ // multiplying with (tau*A^TAv + gam2*v)

    //if (rank == 0)
    //    std::cout << "LBglob = " << LBglob << ", SBglob = " << SBglob << ", red = " << red << std::endl;

    if (v == std::vector<double>(M, 0.0))
        return std::vector<double>(M, 0.0);

    std::vector<double> res(M, 0.0);

    if (red == 0){

        size_t phen_size = 4 * (*dataset).get_mbytes();

        std::vector<double> res_temp(phen_size, 0.0);

        res_temp = (*dataset).Ax(v.data());
        
        res = (*dataset).ATx(res_temp.data());

    } 
    else
    {
        size_t phen_size = 4 * LBglob;

        std::vector<double> res_temp(phen_size, 0.0);

        res_temp = (*dataset).Ax(v.data(), SBglob, LBglob);

        //if (rank == 0)
        //    std::cout << "[lmmse_mult] after Ax" << std::endl;

        res = (*dataset).ATx(res_temp.data(), SBglob, LBglob);

        //if (rank == 0)
        //    std::cout << "[lmmse_mult] after ATx" << std::endl;
    }

    for (int i = 0; i < M; i++){
        res[i] *= tau;
        res[i] += gam2 * v[i];
    }

    return res;
}

std::vector<double> vamp::precondCG_solver(std::vector<double> v, double tau, int denoiser, data* dataset, int red){

    // we start with approximation x0 = 0

    std::vector<double> mu(M, 0.0);

    return precondCG_solver(v, mu, tau, denoiser, dataset, red);

}

std::vector<double> vamp::precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int denoiser, data* dataset, int red){

    // tau = gamw
    // preconditioning part

    std::vector<double> diag(M, 1.0);

    for (int j=0; j<M; j++)
        diag[j] = tau * (N-1) / N + gam2;
         
    std::vector<double> mu = mu_start;
    std::vector<double> d;
    std::vector<double> r = lmmse_mult(mu, tau, dataset, red);

    for (int i0=0; i0<M; i0++)
        r[i0] = v[i0] - r[i0];
    
    std::vector<double> z(M, 0.0);

    //for (int j=0; j<M; j++)
    //    z[j] = r[j] / diag[j];

    std::transform (r.begin(), r.end(), diag.begin(), z.begin(), std::divides<double>());

    std::vector<double> p = z;
    std::vector<double> Apalpha(M, 0.0);
    std::vector<double> palpha(M, 0.0);

    double alpha, beta;
    double prev_onsager = 0;

    for (int i = 0; i < CG_max_iter; i++){

        // d = A*p

        d = lmmse_mult(p, tau, dataset, red);

        alpha = inner_prod(r, z, 1) / inner_prod(d, p, 1);
        
        for (int j = 0; j < M; j++)
            palpha[j] = alpha * p[j];

        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());

        if (denoiser == 0){

            double onsager = gam2 * inner_prod(v, mu, 1);

            double rel_err;

            if (onsager != 0)
                rel_err = abs( (onsager - prev_onsager) / onsager ); 
            else
                rel_err = 1;

            if (rel_err < 1e-8)
                break;

            prev_onsager = onsager;

            if (rank == 0)
                std::cout << "[CG onsager] it = " << i << ": relative error for onsager is " << std::setprecision(10) << rel_err << std::endl;

        }

        for (int j = 0; j < p.size(); j++)
            Apalpha[j] = d[j] * alpha;

        beta = pow(inner_prod(r, z, 1), -1);

        std::transform (r.begin(), r.end(), Apalpha.begin(), r.begin(), std::minus<double>());

        //for (int j=0; j<M; j++)
        //    z[j] = r[j] / diag[j];

        std::transform (r.begin(), r.end(), diag.begin(), z.begin(), std::divides<double>());

        beta *= inner_prod(r, z, 1);

        for (int j = 0; j < p.size(); j++)
            p[j] = z[j] + beta * p[j];

        // stopping criteria
        double norm_v = sqrt(l2_norm2(v, 1));
        double norm_z = sqrt(l2_norm2(z, 1));
        double rel_err = sqrt( l2_norm2(r, 1) ) / norm_v;
        double norm_mu = sqrt( l2_norm2(mu, 1) );
        double err_tol = 1e-5;

        if (rank == 0)
            std::cout << "[CG] it = " << i << ": ||r_it|| / ||RHS|| = " << std::setprecision(10) << rel_err << ", ||x_it|| = " << norm_mu << ", ||z|| / ||RHS|| = " <<  norm_z /  norm_v << std::endl;

        if (rel_err < err_tol) 
            break;
    }
    if (denoiser == 1)
        mu_CG_last = mu;

    return mu;
 }


void vamp::err_measures(data *dataset, int ind){

    double scale = 1.0 / (double) N;
    
    // correlation
    if (ind == 1){

        double corr = inner_prod(x1_hat, true_signal, 1) / sqrt( l2_norm2(x1_hat, 1) * l2_norm2(true_signal, 1) );

        if ( rank == 0 )
            std::cout << "correlation x1_hat = " << corr << std::endl;  

        double l2_norm2_x1_hat = l2_norm2(x1_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);

    }
    else if (ind == 2){

        double corr_2 = inner_prod(x2_hat, true_signal, 1) / sqrt( l2_norm2(x2_hat, 1) * l2_norm2(true_signal, 1) );

        if (rank == 0)
            std::cout << "correlation x2_hat = " << corr_2 << std::endl;

        double l2_norm2_x2_hat = l2_norm2(x2_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);
        
    }
    

    // l2 signal error
    std::vector<double> temp(M, 0.0);
    double l2_norm2_xhat;

    if (ind == 1){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x1_hat[i] - true_signal[i];

        l2_norm2_xhat = l2_norm2(x1_hat, 1) * scale;
    }
    else if (ind == 2){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x2_hat[i] - true_signal[i];

        l2_norm2_xhat = l2_norm2(x2_hat, 1) * scale;
    }


    double l2_signal_err = sqrt( l2_norm2(temp, 1) / l2_norm2(true_signal, 1) );
    if (rank == 0)
        std::cout << "l2 signal error = " << l2_signal_err << std::endl;
    
    
    // l2 prediction error
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> tempNest(phen_size, 0.0);
    std::vector<double> tempNtrue(phen_size, 0.0);

    //filtering pheno vector for NAs
    if (redglob == 0){

        y = (*dataset).filter_pheno();

        std::vector<double> Axest;
        if (ind == 1)
            if (z1.size() > 0)
                Axest = z1;
            else
                Axest = (*dataset).Ax(x1_hat.data());
        else if (ind == 2)
        Axest = (*dataset).Ax(x2_hat.data());

        for (int i = 0; i < N; i++){ // N because length(y) = N even though length(Axest) = 4*mbytes
            tempNest[i] = -Axest[i] + y[i];
        }

        double l2_pred_err = sqrt(l2_norm2(tempNest, 0) / l2_norm2(y, 0));

        if (rank == 0)
            std::cout << "l2 prediction error = " << l2_pred_err << std::endl;

        double R2 = 1 - l2_pred_err * l2_pred_err;

        R2trains.push_back(R2);

        if (rank == 0)
            std::cout << "R2 = " << R2 << std::endl;
    }
    else
    {
        y = (*dataset).filter_pheno();
        std::vector<double> y_filt(phen_size, 0.0);

        std::vector<double> Axest;
        if (ind == 1)
            if (z1.size() > 0)
                Axest = z1;
            else
                Axest = (*dataset).Ax(x1_hat.data(), SBglob, LBglob);
        else if (ind == 2)
        Axest = (*dataset).Ax(x2_hat.data(), SBglob, LBglob);

        for (int i = SBglob; i < SBglob + LBglob; i++){ // N because length(y) = N even though length(Axest) = 4*mbytes
            for (int k=0; k<4; k++){
                tempNest[4*i + k] = -Axest[4*i + k] + y[4*i + k];
                y_filt[4*i+k] = y[4*i+k];
            }
        }

        double l2_pred_err = sqrt(l2_norm2(tempNest, 0) / l2_norm2(y_filt, 0));

        if (rank == 0)
            std::cout << "l2 prediction error = " << l2_pred_err << std::endl;

        double R2 = 1 - l2_pred_err * l2_pred_err;

        if (rank == 0)
            std::cout << "R2 = " << R2 << std::endl;
    }
    

    // prior distribution parameters
    if (rank == 0)
        std::cout << "prior variances = ";

    for (int i = 0; i < vars.size(); i++)
        if (rank == 0)
            std::cout << vars[i] << ' ';

    if (rank == 0) {
        std::cout << std::endl;
        std::cout << "prior probabilities = ";
    }

    for (int i = 0; i < probs.size(); i++)
        if (rank == 0)
            std::cout << probs[i] << ' ';

    if (rank == 0){
        std::cout << std::endl;
        std::cout << "gamw = " << gamw << std::endl;
    }
    
}

std::tuple<double, double, double> vamp::state_evo(int ind, double gam_prev, double gam_before, std::vector<double> probs_before, std::vector<double> vars_before, data* dataset){

    //double gam_before, std::vector<double> probs_before, std::vector<double> vars_before
    // damping is not taken into account
    double alpha_bar = 0, eta_bar = 0, gam_bar = 0;

    if (ind == 1){

        std::vector<double> sim_beta = simulate(M, vars, probs);
        std::vector<double> sim_gam_noise = simulate(M, std::vector<double> {1/gam_prev}, std::vector<double> {1});
        std::vector<double> sim_beta_before = simulate(M, vars_before, probs_before);
        std::vector<double> sim_gam_noise_before = simulate(M, std::vector<double> {1/gam_before}, std::vector<double> {1});

        std::vector<double> r1_den(M, 0.0);
        for (int i=0; i<M; i++)
            r1_den[i] = rho * g1d(sim_beta[i] + sim_gam_noise[i], gam_prev) + (1-rho) * g1d(sim_beta_before[i] + sim_gam_noise_before[i], gam_before);

        double alpha_bar_temp = std::accumulate(r1_den.begin(), r1_den.end(), 0.0);
        MPI_Allreduce(&alpha_bar_temp, &alpha_bar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha_bar /= Mt;

        eta_bar = gam_prev / alpha_bar;

        gam_bar = eta_bar - gam_prev;
    }
    else if (ind == 2){

        alpha_bar = g2d_onsager(gam_prev, gamw, dataset); // * gam_prev ;

        eta_bar = gam_prev / alpha_bar;

        gam_bar = eta_bar - gam_prev;
    }

    return {alpha_bar, eta_bar, gam_bar};
}
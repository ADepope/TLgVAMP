#include <mpi.h>
#include <iostream>
#include <cmath> // contains definition of ceil
#include <bits/stdc++.h>  // contains definition of INT_MAX
#include <immintrin.h> // contains definition of _mm_malloc
#include <numeric>

#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"
#include "options.hpp"


int main(int argc, char** argv)
{

    // starting parallel processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);

    const Options opt(argc, argv);

    // retrieving MPI specific information
    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // reading genotype data / phenotype file
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    if (opt.get_run_mode() == "infere"){ 

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);
        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

        double gam1 = 1e-6;
        double gamw;
        if (opt.get_h2() == -1)
            gamw = 2;
        else 
            gamw = 1.0 / (1.0 - opt.get_h2());
        
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);


        std::vector<double> x_est = emvamp.infere(&dataset);

    }
    else if (opt.get_run_mode() == "test") // just analyzing the result on the test data
    {   

        // reading test set
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS = divide_work(Mt_test);
        int M_test = MS[0];
        int S_test = MS[1];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);
        
        std::vector<double> y_test = dataset_test.get_phen();

        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        double maxR2 = -1;
        int maxind = -1;

        if (min_it != -1){
            for (int it = min_it; it <= max_it; it++){
                std::vector<double> x_est;
                std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;
                //if (rank == 0)
                    //std::cout << "est_file_name_it = " << est_file_name_it << std::endl;
                if (end_est_file_name == "bin")
                    x_est = mpi_read_vec_from_file(est_file_name_it, M_test, S_test);
                else
                    x_est = read_vec_from_file(est_file_name_it, M_test, S_test);

                for (int i0 = 0; i0 < x_est.size(); i0++)
                    x_est[i0] *= sqrt( (double) N_test );
                
                std::vector<double> z_test = dataset_test.Ax(x_est.data());

                double l2_pred_err2 = 0;
                for (int i0 = 0; i0 < N_test; i0++){
                    // std::cout << "(y_test-z_test)[" << i0 << "] = " << y_test[i0] - z_test[i0] << std::endl;
                    l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
                }  

                double stdev = calc_stdev(y_test);
                double R2 = 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() );
                if (rank == 0){
                    std::cout <<  R2 << ", ";
                }

                if (R2 > maxR2){
                    maxR2 = R2;
                    maxind = it;
                }
            }

            if (rank == 0){
                std::cout << std::endl << "max R2 = " << maxR2 << std::endl;
                std::cout << std::endl << "max ind = " << maxind << std::endl;
            }

        }
        else 
        {
            std::vector<double> x_est;
            if (rank == 0)
                std::cout << "est_file_name = " << est_file_name << std::endl;
            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name, M_test, S_test);
            else
                x_est = read_vec_from_file(est_file_name, M_test, S_test);

            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );

            std::vector<double> z_test = dataset_test.Ax(x_est.data());

            double l2_pred_err2 = 0;
            for (int i0 = 0; i0 < N_test; i0++){
                l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
            }  

            double stdev = calc_stdev(y_test);
            if (rank == 0){
                std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
                std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
                std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
            }
        }
        
    }
    else if (opt.get_run_mode() == "both")
    {

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];
  
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        std::string phenfp = (opt.get_phen_files())[0]; // currently only one phenotype file is supported
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);

        double gam1 = 1e-6; 
        double gamw;
        
        if (opt.get_h2() == -1)
            gamw = 2;
        else 
            gamw = 1.0 / (1.0 - opt.get_h2());

        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);
        std::vector<double> x_est = emvamp.infere(&dataset);

        double intercept = dataset.get_intercept();
        double scale = dataset.get_scale();
        if (rank == 0){
            std::cout << "intercept = " << intercept << std::endl;
            std::cout << "scale = " << scale << std::endl;
        }
                
        int final_it = opt.get_iterations();

        // reading test set
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS_test = divide_work(Mt);
        int M_test = MS_test[0];
        int S_test = MS_test[1];

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S, rank, type_data, alpha_scale, bimfp);

        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data());

        for (int i0 = 0; i0 < z_test.size(); i0++)
            z_test[i0] = intercept + scale * z_test[i0];

        std::vector<double> y_test = dataset_test.get_phen();

        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N_test; i0++){
            l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
        }

        double stdev = calc_stdev(y_test);
        if (rank == 0){
            std::cout << std::endl;
            std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
            std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
            std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
        }
    }
    else if (opt.get_run_mode() == "pvals-calc")
    {

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // reading an estimate file 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.rfind(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        double maxR2 = -1;
        int maxind = -1;

        std::vector< std::vector<double> > z1_hats;
        std::vector< std::vector<double> > x1_hats;
        std::vector< std::string > filepaths_out_pvals;
        std::vector< std::string > filepaths_out_pvals_LOCO;
        int store_pvals = opt.get_store_pvals(); // 0 = run LOO and LOCO, 1 = run only LOO, 2 = run only LOCO

        if (min_it != -1){
            for (int it = min_it; it <= max_it; it++){
                std::vector<double> x_est;
                std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;

                if (end_est_file_name == "bin")
                    x_est = mpi_read_vec_from_file(est_file_name_it, M, S);
                else
                    x_est = read_vec_from_file(est_file_name_it, M, S);

                for (int i0 = 0; i0 < x_est.size(); i0++)
                    x_est[i0] *= sqrt( (double) N );

                x1_hats.push_back(x_est);
                
                std::vector<double> z1_hat = dataset.Ax(x_est.data());

                z1_hats.push_back(z1_hat);

                std::string filepath_out_pvals = opt.get_out_dir() + opt.get_out_name() + "_it_" + std::to_string(it);
                std::string filepath_out_pvals_LOCO = opt.get_out_dir() + opt.get_out_name() + "_it_" + std::to_string(it);
                if (rank == 0)
                    std::cout << "filepath_out_pvals = " << filepath_out_pvals << std::endl;
                if (rank == 0)
                    std::cout << "filepath_out_pvals_LOCO = " << filepath_out_pvals_LOCO << std::endl;
                filepaths_out_pvals.push_back(filepath_out_pvals + "_pvals.bin");
                filepaths_out_pvals_LOCO.push_back(filepath_out_pvals_LOCO + "_pvals_LOCO.bin");
            }
        }
        else
        {

            std::string est_file_name = opt.get_estimate_file();
            int pos_dot = est_file_name.rfind(".");
            std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
            if (rank == 0)
                std::cout << "end_est_file_name = " << end_est_file_name << std::endl;

            std::vector<double> x_est;
            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name, M, S);
            else
                x_est = read_vec_from_file(est_file_name, M, S);

            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N );

            x1_hats.push_back(x_est);

            std::vector<double> z1_hat = dataset.Ax(x_est.data());

            z1_hats.push_back(z1_hat);

            std::string filepath_out_pvals = opt.get_out_dir() + opt.get_out_name() + "_pvals.bin";
            std::string filepath_out_pvals_LOCO = opt.get_out_dir() + opt.get_out_name() + "_pvals_LOCO.bin";
            if (rank == 0)
                std::cout << "filepath_out_pvals = " << filepath_out_pvals << std::endl;
            if (rank == 0)
                std::cout << "filepath_out_pvals_LOCO = " << filepath_out_pvals_LOCO << std::endl;
            filepaths_out_pvals.push_back(filepath_out_pvals);
            filepaths_out_pvals_LOCO.push_back(filepath_out_pvals_LOCO);
        }


        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //   obtaining p-values 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        std::vector<double> y =  dataset.filter_pheno();

        if (store_pvals == 0 || store_pvals == 1)
            std::vector< std::vector<double> > pvals = dataset.pvals_calc(z1_hats, y, x1_hats, filepaths_out_pvals);

        // calculating p-values using LOCO method, if .bim file is specified
        if (dataset.get_bimfp() != "" && (store_pvals == 0 || store_pvals == 2) )
            std::vector< std::vector<double> > pvals_LOCO = dataset.pvals_calc_LOCO(z1_hats, y, x1_hats, filepaths_out_pvals_LOCO);
    }
    else if (opt.get_run_mode() == "restart"){ 

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);
        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double gam1 = opt.get_gam1_init();
        double gamw = opt.get_gamw_init();
        
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);

        std::vector<double> x_est = emvamp.infere(&dataset);

    }
    else if (opt.get_run_mode() == "predict"){ 

    // reading test set
    const std::string bedfp_test = opt.get_bed_file_test();
    
    int N_test = opt.get_N_test();
    int Mt_test = opt.get_Mt_test();
    std::vector<double> MS = divide_work(Mt_test);
    int M_test = MS[0];
    int S_test = MS[1];
    std::string type_data = "bed";
    double alpha_scale = opt.get_alpha_scale();
    std::string bimfp = opt.get_bim_file();

    std::vector<double> phen(N_test, 0.0);
    data dataset_test(phen, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);

    std::string est_file_name = opt.get_estimate_file();
    int pos_dot = est_file_name.find(".");
    std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
    if (rank == 0)
        std::cout << "est_file_name = " << est_file_name << std::endl;

    int pos_it = est_file_name.rfind("temp");
    std::vector<int> iter_range = opt.get_test_iter_range();
    int min_it = iter_range[0];
    int max_it = iter_range[1];
    if (rank == 0)
        std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

    std::vector< std::vector<double> > z_test_mat;

    if (min_it != -1){
        for (int it = min_it; it <= max_it; it++){
            std::vector<double> x_est;
            std::string est_file_name_it = est_file_name.substr(0, pos_it) + "temp_" + std::to_string(it) + "_" + std::to_string(it) + "_gibbs_est." + end_est_file_name;
            if (rank == 0)
                std::cout << "est_file_name_it = " << est_file_name_it << std::endl;
            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name_it, M_test, S_test);
            else
                x_est = read_vec_from_file(est_file_name_it, M_test, S_test);

            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );
            
            std::vector<double> z_test = dataset_test.Ax(x_est.data());
            z_test_mat.push_back(z_test);            
        }

        for (int i=0; i<N_test; i++){
            std::vector<double> row(max_it-min_it+1, 0.0);
            for (int it = min_it; it <= max_it; it++){ // min_it >= 1
                row[it-min_it] = z_test_mat[it-min_it][i];
            }
            std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_predict_" + std::to_string(i) + ".csv";
            if (rank == 0){
                std::cout << "filepath_out = " << filepath_out << std::endl;
                store_vec_to_file(filepath_out, row);
                }
            }
        }
    }
    else if (opt.get_run_mode() == "predict_single"){ 

    // reading test set
    const std::string bedfp_test = opt.get_bed_file_test();
   
    int N_test = opt.get_N_test();
    int Mt_test = opt.get_Mt_test();
    std::vector<double> MS = divide_work(Mt_test);
    int M_test = MS[0];
    int S_test = MS[1];
    std::string type_data = "bed";
    double alpha_scale = opt.get_alpha_scale();
    std::string bimfp = opt.get_bim_file();

    std::vector<double> y(N_test, 0.0);

    data dataset_test(y, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);

    std::string est_file_name = opt.get_estimate_file();
    int pos_dot = est_file_name.find(".");
    std::string end_est_file_name = est_file_name.substr(pos_dot + 1);

    std::vector<double> x_est;
    if (end_est_file_name == "bin")
        x_est = mpi_read_vec_from_file(est_file_name, M_test, S_test);
    else
        x_est = read_vec_from_file(est_file_name, M_test, S_test);

    for (int i0 = 0; i0 < x_est.size(); i0++)
        x_est[i0] *= sqrt( (double) N_test );
        
    std::vector<double> z_test = dataset_test.Ax(x_est.data());           

    std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_predict.csv";
    if (rank == 0){
            std::cout << "filepath_out = " << filepath_out << std::endl;
            store_vec_to_file(filepath_out, z_test);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
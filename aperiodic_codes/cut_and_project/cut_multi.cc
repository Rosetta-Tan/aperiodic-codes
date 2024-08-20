#include <vector>
#include <sstream>
#include <chrono>
#include <thread>
#include "Highs.h"
#include "cnpy.h"

using namespace std;

vector<size_t> partition(size_t M , size_t m) {
    auto part = vector<size_t>(m+1,0);
    for(auto i = 1lu ; i <= m ; ++i) part.at(i) = (M*i)/m;
    return part;
    }

void cut_work(HighsModel model,
              const double* orth_pts ,
              vector<vector<size_t> >& cut_ind_multi ,
              vector<size_t> const& part ,
              size_t num) {
    const auto d = model.lp_.num_row_-1lu;
    vector<double> pt(d+1); pt[d] = 1.0;
    vector<size_t>& cur_ind = cut_ind_multi[num];
    Highs highs;
    HighsBasis basis;

    highs.setOptionValue("solver" , "simplex");
    highs.setOptionValue("presolve" , "off");
    highs.setOptionValue("parallel" , "off");
    highs.setOptionValue("output_flag" , "false");
    highs.setOptionValue("log_to_console" , "false");

    for(auto i = part[num] ; i < part[num+1] ; ++i) {
        for(auto j = 0lu ; j < d ; ++j) pt[j] = orth_pts[i*d+j];
        model.lp_.row_lower_ = pt;
        model.lp_.row_upper_ = pt;

        basis = highs.getBasis();
        highs.passModel(model);
        highs.setBasis(basis);
        highs.run();

        if(highs.getModelStatus() == HighsModelStatus::kOptimal) cur_ind.push_back(i);
        }

    return;
    }

int main(int argc, char *argv[]) {
    if(argc != 3) { cerr << "Usage: " << argv[0] << " file_base num_threads" << endl; return 1; }

    // Load numpy data
    ostringstream df_name; df_name << argv[1] << "_cut.npz";
    auto orth_data = cnpy::npz_load(df_name.str());
    auto orth_pts_arr = orth_data["orth_pts"] , orth_wdw_arr = orth_data["orth_window"];
    double *orth_pts = orth_pts_arr.data<double>() , *orth_wdw = orth_wdw_arr.data<double>();
    assert(orth_pts_arr.shape.size() == 2 && orth_wdw_arr.shape.size() == 2 &&
           orth_pts_arr.shape[1] == orth_wdw_arr.shape[0]);  

    const auto N   = orth_pts_arr.shape[0],
               d   = orth_wdw_arr.shape[0],
               n   = orth_wdw_arr.shape[1],
               nTh = stoul(argv[2]);
    
    // Set up linear programming model for threads
    auto start_vec = vector<int>(d+2), 
         index_vec = vector<int>(n*(d+1));
    auto value_vec = vector<double>(orth_wdw, orth_wdw+n*d),
         zeros_vec = vector<double>(n),
         ones_vec  = vector<double>(n,1.0);
    value_vec.insert(value_vec.end(),ones_vec.begin(),ones_vec.end());

    for(auto i = 0lu ; i <= d ; ++i) {
        start_vec.at(i) = static_cast<int>(i*n);
        for(auto j = 0lu ; j < n ; ++j)
            index_vec.at(i*n+j) = static_cast<int>(j);
        }
    start_vec.at(d+1) = static_cast<int>(n*(d+1));
    
    HighsModel model;
    model.lp_.num_col_ = n;
    model.lp_.num_row_ = d+1;
    model.lp_.col_cost_  = zeros_vec;
    model.lp_.col_lower_ = zeros_vec;
    model.lp_.col_upper_ = ones_vec;
    model.lp_.a_matrix_.format_ = MatrixFormat::kRowwise;
    model.lp_.a_matrix_.start_  = start_vec;
    model.lp_.a_matrix_.index_  = index_vec;
    model.lp_.a_matrix_.value_  = value_vec;

    // Parallel cut
    vector<vector<size_t>> cut_ind_multi(nTh);
    vector<thread> threads(nTh);
    const auto part = partition(N,nTh);

    //auto t0 = chrono::high_resolution_clock::now();
    for(auto i = 0lu ; i < nTh ; ++i) {
        cut_ind_multi[i].reserve(N/nTh);
        threads[i] = thread(cut_work,model,orth_pts,ref(cut_ind_multi),ref(part),i);
        }
    for(auto i = 0lu ; i < nTh ; ++i) threads[i].join();
    //auto t1 = chrono::high_resolution_clock::now();
    //cerr << "cut duration " << chrono::duration_cast<chrono::duration<double>>(t1 - t0) << endl;

    // Save to npy file
    auto cut_ind = vector<size_t>();
    for(auto const& v : cut_ind_multi) cut_ind.insert(cut_ind.end(),v.begin(),v.end());
    
    ostringstream pf_name; pf_name << argv[1] << "_ind.npy";
    cnpy::npy_save(pf_name.str(),&cut_ind[0],{cut_ind.size()},"w");

    return 0;
    }

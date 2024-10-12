#include <vector>
#include <thread>
#include "Highs.h"

using std::vector;
using std::thread;

vector<int> partition(int M , int m) {
    auto part = vector<int>(m+1,0);
    for(auto i = 1 ; i <= m ; ++i) part.at(i) = (M*i)/m;
    return part;
    }

void cut_work(HighsModel model,
              const double* orth_pts ,
              int *cut_mask ,
              vector<int> const& part ,
              int num) {
    const auto d = model.lp_.num_row_-1lu;
    vector<double> pt(d+1); pt[d] = 1.0;
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

        if(highs.getModelStatus() == HighsModelStatus::kOptimal)
            cut_mask[i] = 1;
        }

    return;
    }

extern "C" {
void cut(const double *orth_pts , const double *orth_wdw , int *cut_mask , int N , int d , int n , int nTh) {
    // Set up linear programming model for threads
    auto start_vec = vector<int>(d+2), 
         index_vec = vector<int>(n*(d+1));
    auto value_vec = vector<double>(orth_wdw, orth_wdw+n*d),
         zeros_vec = vector<double>(n),
         ones_vec  = vector<double>(n,1.0);
    value_vec.insert(value_vec.end(),ones_vec.begin(),ones_vec.end());

    for(auto i = 0 ; i <= d ; ++i) {
        start_vec.at(i) = i*n;
        for(auto j = 0 ; j < n ; ++j)
            index_vec.at(i*n+j) = j;
        }
    start_vec.at(d+1) = n*(d+1);

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
    vector<thread> threads(nTh);
    const auto part = partition(N,nTh);

    for(auto i = 0 ; i < nTh ; ++i)
        threads[i] = thread(cut_work,model,orth_pts,cut_mask,ref(part),i);

    for(auto i = 0 ; i < nTh ; ++i)
        threads[i].join();

    return;
    }
}

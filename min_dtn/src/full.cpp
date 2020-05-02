// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: CL (no real functionality - binder test only)
//

#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

//#include "min_dtn.hpp" 
#include "Eigen/Core"
// #include <vector>
// #include "Eigen/StdVector"
#include "ceres/ceres.h"
#include "ceres/loss_function.h"
#include "glog/logging.h"
#include <stdexcept>

// std
using std::shared_ptr;
// ceres 
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using ceres::ScaledLoss;
using ceres::TAKE_OWNERSHIP;
// Eigen
using Eigen::Dynamic;
using Eigen::RowMajor;
typedef std::complex<double> Complex;
typedef Eigen::Matrix<Complex,Dynamic,1> ComplexVector;
typedef Eigen::Matrix<double,Dynamic,1> RealVector;
typedef Eigen::Matrix<Complex,Dynamic,Dynamic,RowMajor> ComplexMatrix;
typedef Eigen::Matrix<double,Dynamic,Dynamic,RowMajor> RealMatrix;
typedef Eigen::Matrix< std::complex<const double> ,Dynamic,Dynamic,RowMajor> ConstComplexMatrix;



void eval_dtn_fct(Eigen::Ref<ComplexMatrix> Lone,
	          Eigen::Ref<ComplexMatrix> Ltwo,
		  Eigen::Ref<RealVector> lam,
		  Eigen::Ref<ComplexVector> val
		) 
{
  int N = Lone.cols();
  
  ComplexMatrix S(N,N); 
  ComplexMatrix SE;
  ComplexMatrix SE_inv;
  
  if(N > 1) { 
    SE.resize(N-1,N-1);
  }

  for(int k=0; k < lam.size(); k++) {
 
    for (int i = 0; i < N; i++){
      for(int j=  0; j < N; j++){
        S(i,j) = Lone(i,j) + lam(k)*Ltwo(i,j);
	if (N > 1 && i > 0 && j > 0) {
	  SE(i-1,j-1) = S(i,j);
	}
      }
    }
      
    if( N > 1) {
      SE_inv = SE.inverse();
    }

    Complex zeta_approx = S(0,0);
    if (N > 1) {
      for (int i=1;i<N;i++){
        for(int j=1;j<N;j++){
          zeta_approx -= S(0,i)*SE_inv(i-1,j-1)*S(j,0);
	  }
	}
      }
      val(k) = zeta_approx;	
  }
      
}




namespace py = pybind11;

PYBIND11_MODULE(min_dtn, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: cmake_example
        .. autosummary::
           :toctree: _generate
    )pbdoc";

    m.def("eval_dtn_fct",[] ( Eigen::Ref<ComplexMatrix> Lone,
                              Eigen::Ref<ComplexMatrix> Ltwo,
                              Eigen::Ref<RealVector> lam,
                              Eigen::Ref<ComplexVector> val
            ) {
                           eval_dtn_fct(Lone,Ltwo,lam,val);
                         }, 
          py::arg("Lone"),py::arg("Ltwo"),py::arg("lam"),py::arg("val")
      ); 
      
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}


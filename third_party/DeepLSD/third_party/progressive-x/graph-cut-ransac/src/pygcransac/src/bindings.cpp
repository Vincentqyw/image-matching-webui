#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include "gcransac_python.h"


namespace py = pybind11;


py::tuple findRigidTransform(
	py::array_t<double>  x1y1z1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  probabilities_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler,
	int neighborhood,
	double neighborhood_size)
{
	py::buffer_info buf1 = x1y1z1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 3) {
		throw std::invalid_argument("x1y1z1 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTS < 3) {
		throw std::invalid_argument("x1y1z1 should be an array with dims [n,3], n>=3");
	}
	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1z1 and x2y2z2 should be the same size");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1z1;
	x1y1z1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

	std::vector<double> pose(16);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findRigidTransform_(
		x1y1z1,
		x2y2z2,
		probabilities,
		inliers,
		pose,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters,
		use_sprt,
		min_inlier_ratio_for_sprt,
		sampler,
		neighborhood,
		neighborhood_size);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];
	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}
	py::array_t<double> pose_ = py::array_t<double>({ 4,4 });
	py::buffer_info buf2 = pose_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 16; i++)
		ptr2[i] = pose[i];
	return py::make_tuple(pose_, inliers_);
}


py::tuple find6DPose(
	py::array_t<double>  x1y1_,
	py::array_t<double>  x2y2z2_,
	py::array_t<double>  probabilities_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler,
	int neighborhood,
	double neighborhood_size)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
	}
	if (NUM_TENTS < 7) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=7");
	}
	py::buffer_info buf1a = x2y2z2_.request();
	size_t NUM_TENTSa = buf1a.shape[0];
	size_t DIMa = buf1a.shape[1];

	if (DIMa != 3) {
		throw std::invalid_argument("x2y2z2 should be an array with dims [n,3], n>=3");
	}
	if (NUM_TENTSa != NUM_TENTS) {
		throw std::invalid_argument("x1y1 and x2y2z2 should be the same size");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

	double *ptr1a = (double *)buf1a.ptr;
	std::vector<double> x2y2z2;
	x2y2z2.assign(ptr1a, ptr1a + buf1a.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

	std::vector<double> pose(12);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = find6DPose_(
		x1y1,
		x2y2z2,
		probabilities,
		inliers,
		pose,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters,
		use_sprt,
		min_inlier_ratio_for_sprt,
		sampler,
		neighborhood,
		neighborhood_size);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];
	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}
	py::array_t<double> pose_ = py::array_t<double>({ 3,4 });
	py::buffer_info buf2 = pose_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 12; i++)
		ptr2[i] = pose[i];
	return py::make_tuple(pose_, inliers_);
}

py::tuple findFundamentalMatrix(
	py::array_t<double>  correspondences_,
	int h1, int w1, int h2, int w2,
	py::array_t<double>  probabilities_,
	double threshold,
	double conf,
	double spatial_coherence_weight,
	int max_iters,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler,
	int neighborhood,
	double neighborhood_size)
{
	py::buffer_info buf1 = correspondences_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 4) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
	}
	if (NUM_TENTS < 7) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,4], n>=7");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> correspondences;
	correspondences.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

	std::vector<double> F(9);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findFundamentalMatrix_(
		correspondences,
		probabilities,
		inliers,
		F,
		h1, w1, h2, w2,
		spatial_coherence_weight,
		threshold,
		conf,
		max_iters,
		use_sprt,
		min_inlier_ratio_for_sprt,
		sampler,
		neighborhood,
		neighborhood_size);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];
	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}
	py::array_t<double> F_ = py::array_t<double>({ 3,3 });
	py::buffer_info buf2 = F_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 9; i++)
		ptr2[i] = F[i];
	return py::make_tuple(F_, inliers_);
}


py::tuple findLine2D(py::array_t<double>  x1y1_,
	int w1, int h1,
	py::array_t<double>  probabilities_,
	double threshold,
	double conf,
	int max_iters,
	double spatial_coherence_weight,
	bool use_sprt,
	double min_inlier_ratio_for_sprt,
	int sampler,
	int neighborhood,
	double neighborhood_size)
{
	py::buffer_info buf1 = x1y1_.request();
	size_t NUM_TENTS = buf1.shape[0];
	size_t DIM = buf1.shape[1];

	if (DIM != 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}
	if (NUM_TENTS < 2) {
		throw std::invalid_argument("x1y1 should be an array with dims [n,2], n>=2");
	}

	double *ptr1 = (double *)buf1.ptr;
	std::vector<double> x1y1;
	x1y1.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

	std::vector<double> linemodel(3);
	std::vector<bool> inliers(NUM_TENTS);

	int num_inl = findLine2D_(x1y1,
		probabilities,
		inliers,
		linemodel,
		w1, h1,
		threshold,
		conf,
		max_iters,
		spatial_coherence_weight,
		use_sprt,
		min_inlier_ratio_for_sprt,
		sampler,
		neighborhood,
		neighborhood_size);

	py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
	py::buffer_info buf3 = inliers_.request();
	bool *ptr3 = (bool *)buf3.ptr;
	for (size_t i = 0; i < NUM_TENTS; i++)
		ptr3[i] = inliers[i];
	if (num_inl == 0) {
		return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
	}
	py::array_t<double> F_ = py::array_t<double>({ 3 });
	py::buffer_info buf2 = F_.request();
	double *ptr2 = (double *)buf2.ptr;
	for (size_t i = 0; i < 3; i++)
		ptr2[i] = linemodel[i];
	return py::make_tuple(F_, inliers_);
}


py::tuple findEssentialMatrix(py::array_t<double>  correspondences_,
                                py::array_t<double>  K1_,
                                py::array_t<double>  K2_,
                                int h1, int w1, int h2, int w2,
    					 		py::array_t<double>  probabilities_,
								double threshold,
								double conf,
								double spatial_coherence_weight,
								int max_iters,
								bool use_sprt,
								double min_inlier_ratio_for_sprt,
								int sampler,
								int neighborhood,
								double neighborhood_size)
{
    py::buffer_info buf1 = correspondences_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=5" );
    }
    if (NUM_TENTS < 5) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=5");
    }

    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> corrs;
    corrs.assign(ptr1, ptr1 + buf1.size);

    py::buffer_info K1_buf = K1_.request();
    size_t three_a = K1_buf.shape[0];
    size_t three_b = K1_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument( "K1 shape should be [3x3]");
    }
    double *ptr1_k = (double *) K1_buf.ptr;
    std::vector<double> K1;
    K1.assign(ptr1_k, ptr1_k + K1_buf.size);

    py::buffer_info K2_buf = K2_.request();
    three_a = K2_buf.shape[0];
    three_b = K2_buf.shape[1];

    if ((three_a != 3) || (three_b != 3)) {
        throw std::invalid_argument( "K2 shape should be [3x3]");
    }
    double *ptr2_k = (double *) K2_buf.ptr;
    std::vector<double> K2;
    K2.assign(ptr2_k, ptr2_k + K2_buf.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }

    std::vector<double> F(9);
    std::vector<bool> inliers(NUM_TENTS);

    int num_inl = findEssentialMatrix_(
							corrs,
							probabilities,
                           	inliers,
                           	F, K1, K2,
                           	h1, w1, h2, w2,
						   	spatial_coherence_weight,
                           	threshold,
						   	conf,
						   	max_iters,
						   	use_sprt,
							min_inlier_ratio_for_sprt,
							sampler,
							neighborhood,
							neighborhood_size);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];
    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> F_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = F_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = F[i];
    return py::make_tuple(F_,inliers_);
}

py::tuple findHomography(py::array_t<double>  correspondences_,
                         int h1, int w1, int h2, int w2,
    					 py::array_t<double>  probabilities_,
                         double threshold,
                         double conf,
							double spatial_coherence_weight,
							int max_iters,
							bool use_sprt,
							double min_inlier_ratio_for_sprt,
							int sampler,
							int neighborhood,
							double neighborhood_size,
							bool use_space_partitioning)
{
    py::buffer_info buf1 = correspondences_.request();
    size_t NUM_TENTS = buf1.shape[0];
    size_t DIM = buf1.shape[1];

    if (DIM != 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=4" );
    }
    if (NUM_TENTS < 4) {
        throw std::invalid_argument( "correspondences should be an array with dims [n,4], n>=4");
    }

    double *ptr1 = (double *) buf1.ptr;
    std::vector<double> corrs;
    corrs.assign(ptr1, ptr1 + buf1.size);

    std::vector<double> probabilities;
    if (sampler == 3 || sampler == 4)
    {
        py::buffer_info buf_prob = probabilities_.request();
        double* ptr_prob = (double*)buf_prob.ptr;
        probabilities.assign(ptr_prob, ptr_prob + buf_prob.size);        
    }
	
    std::vector<double> H(9);
    std::vector<bool> inliers(NUM_TENTS);

    int num_inl = findHomography_(
					corrs,
					probabilities,
                    inliers,
                    H,
                    h1, w1,h2,w2,
					spatial_coherence_weight,
                    threshold,
                    conf,
                    max_iters,
					use_sprt,
					min_inlier_ratio_for_sprt,
					sampler,
					neighborhood,
					neighborhood_size,
					use_space_partitioning);

    py::array_t<bool> inliers_ = py::array_t<bool>(NUM_TENTS);
    py::buffer_info buf3 = inliers_.request();
    bool *ptr3 = (bool *)buf3.ptr;
    for (size_t i = 0; i < NUM_TENTS; i++)
        ptr3[i] = inliers[i];

    if (num_inl  == 0){
        return py::make_tuple(pybind11::cast<pybind11::none>(Py_None),inliers_);
    }
    py::array_t<double> H_ = py::array_t<double>({3,3});
    py::buffer_info buf2 = H_.request();
    double *ptr2 = (double *)buf2.ptr;
    for (size_t i = 0; i < 9; i++)
        ptr2[i] = H[i];

    return py::make_tuple(H_,inliers_);
                         }
PYBIND11_PLUGIN(pygcransac) {

    py::module m("pygcransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pygcransac
        .. autosummary::
           :toctree: _generate

           findFundamentalMatrix,
			findLine2D,
			findHomography,
		   find6DPose,
		   findEssentialMatrix,
		   findRigidTransform,

    )doc");

	m.def("findFundamentalMatrix", &findFundamentalMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
		py::arg("h1"),
		py::arg("w1"),
		py::arg("h2"),
		py::arg("w2"),
        py::arg("probabilities"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000,
		py::arg("use_sprt") = true,
		py::arg("min_inlier_ratio_for_sprt") = 0.1,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 20.0);

		m.def("findLine2D", &findLine2D, R"doc(some doc)doc",
			py::arg("x1y1"),
			py::arg("w1"),
			py::arg("h1"),
        	py::arg("probabilities"),
			py::arg("threshold") = 1.0,
			py::arg("conf") = 0.99,
			py::arg("max_iters") = 10000,
			py::arg("spatial_coherence_weight") = 0.975,
			py::arg("use_sprt") = false,
			py::arg("min_inlier_ratio_for_sprt") = 0.1,
			py::arg("sampler") = 0,
			py::arg("neighborhood") = 0,
			py::arg("neighborhood_size") = 20.0);


	m.def("findRigidTransform", &findRigidTransform, R"doc(some doc)doc",
		py::arg("x1y1z1"),
		py::arg("x2y2z2"),
        py::arg("probabilities"),
		py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000,
		py::arg("use_sprt") = true,
		py::arg("min_inlier_ratio_for_sprt") = 0.1,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 20.0);

	m.def("find6DPose", &find6DPose, R"doc(some doc)doc",
		py::arg("x1y1"),
		py::arg("x2y2z2"),
        py::arg("probabilities"),
		py::arg("threshold") = 0.001,
		py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
		py::arg("max_iters") = 10000,
		py::arg("use_sprt") = true,
		py::arg("min_inlier_ratio_for_sprt") = 0.1,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 8.0);

    m.def("findEssentialMatrix", &findEssentialMatrix, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("K1"),
        py::arg("K2"),
        py::arg("h1"),
        py::arg("w1"),
        py::arg("h2"),
        py::arg("w2"),
        py::arg("probabilities"),
        py::arg("threshold") = 1.0,
        py::arg("conf") = 0.99,
		py::arg("spatial_coherence_weight") = 0.975,
        py::arg("max_iters") = 10000,
		py::arg("use_sprt") = true,
		py::arg("min_inlier_ratio_for_sprt") = 0.1,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 8.0);

  m.def("findHomography", &findHomography, R"doc(some doc)doc",
        py::arg("correspondences"),
        py::arg("h1"),
        py::arg("w1"),
        py::arg("h2"),
        py::arg("w2"),
        py::arg("probabilities"),
        py::arg("threshold") = 1.0,
		py::arg("conf") = 0.99,
        py::arg("spatial_coherence_weight") = 0.975,
        py::arg("max_iters") = 10000,
	  	py::arg("use_sprt") = true,
	  	py::arg("min_inlier_ratio_for_sprt") = 0.1,
		py::arg("sampler") = 1,
		py::arg("neighborhood") = 0,
		py::arg("neighborhood_size") = 4.0,
		py::arg("use_space_partitioning") = true);

  return m.ptr();
}


#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <random>
#include "hest.h"

struct Instance {
    std::vector<hest::LineSegment> endpoints1;
    std::vector<hest::LineSegment> endpoints2;
    Eigen::Matrix3d homography_gt;
    std::vector<Eigen::Vector2d> corners1;
    std::vector<Eigen::Vector2d> corners2;
};

struct ProblemOptions {
    double image_size = 1000.0;
    double corner_noise = 100.0;
    int n_lines = 100;            
    int n_outliers = 50;        
    double noise_level = 0.5; // In pixels 
};

double computeHomographyError(const Eigen::Matrix3d &H, const Instance &instance) {
    double err = 0.0;
    for(int i = 0; i < 4; ++i) {
        Eigen::Vector2d c = (H * instance.corners2[i].homogeneous()).hnormalized();
        err += (c - instance.corners1[i]).norm();
    }
    return err / 4.0;
}


Instance generateInstance(const ProblemOptions &options) {

    // Random generators
    std::default_random_engine rng;
    std::uniform_real_distribution<double> coord_gen(0,options.image_size);        
    std::normal_distribution<double> gaussian_noise_gen(0.0, 1.0);    

    // Generate GT homography
    Instance instance;

    instance.corners1 = {
        {0.0, 0.0},
        {options.image_size, 0.0},
        {0.0, options.image_size},
        {options.image_size, options.image_size}
    };

    instance.corners2 = instance.corners1;
    for(Eigen::Vector2d &c : instance.corners2) {
        c.x() += options.corner_noise * gaussian_noise_gen(rng);
        c.y() += options.corner_noise * gaussian_noise_gen(rng);
    }

    instance.homography_gt = hest::estimateHomographyPoints(instance.corners1, instance.corners2);

    instance.endpoints1.resize(options.n_lines);
    instance.endpoints2.resize(options.n_lines);

    for(int i = 0; i < options.n_lines; ++i) {
        instance.endpoints2[i].p1 << coord_gen(rng), coord_gen(rng);
        instance.endpoints2[i].p2 << coord_gen(rng), coord_gen(rng);

        instance.endpoints1[i].p1 << (instance.homography_gt * instance.endpoints2[i].p1.homogeneous()).hnormalized();
        instance.endpoints1[i].p2 << (instance.homography_gt * instance.endpoints2[i].p2.homogeneous()).hnormalized();
    }

    // add noise
    for(int i = 0; i < options.n_lines; ++i) {
        instance.endpoints1[i].p1 += options.noise_level * Eigen::Vector2d(gaussian_noise_gen(rng),gaussian_noise_gen(rng));
        instance.endpoints1[i].p2 += options.noise_level * Eigen::Vector2d(gaussian_noise_gen(rng),gaussian_noise_gen(rng));
        instance.endpoints2[i].p1 += options.noise_level * Eigen::Vector2d(gaussian_noise_gen(rng),gaussian_noise_gen(rng));
        instance.endpoints2[i].p2 += options.noise_level * Eigen::Vector2d(gaussian_noise_gen(rng),gaussian_noise_gen(rng));
    }

    // add outliers
    std::vector<int> ind;
    for(int i = 0; i < options.n_lines; ++i) {
        ind.push_back(i);
    }
    std::shuffle(ind.begin(),ind.end(), rng);
    for(int i = 0; i < options.n_outliers; ++i) {
        instance.endpoints2[ind[i]].p1 << coord_gen(rng), coord_gen(rng);
        instance.endpoints2[ind[i]].p2 << coord_gen(rng), coord_gen(rng);
    }



    return instance;
}





int main() {

    ProblemOptions opt;
    opt.n_lines = 100;
    opt.n_outliers = 50;
    
    Instance instance = generateInstance(opt);

    double tol_px = 5.0;
    std::vector<int> inliers;
    Eigen::Matrix3d H = hest::ransacHomography(instance.endpoints1, instance.endpoints2, tol_px, &inliers);

    std::cout << " Found homography with " << inliers.size() << " / " << opt.n_lines << " inliers\n";
    
    double err = computeHomographyError(H, instance);

    std::cout << "H = \n" << H <<"\n";
    std::cout << "H_gt = \n" << instance.homography_gt <<"\n";

    std::cout << "err = " << err << "\n";


}
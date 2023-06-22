#ifndef POSELIB_ROBUST_LOSS_H_
#define POSELIB_ROBUST_LOSS_H_

namespace pose_lib {

// Robust loss functions
class TrivialLoss {
  public:
    TrivialLoss(double) {} // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    double loss(double r2) const {
        return r2;
    }
    double weight(double r2) const {
        return 1.0;
    }
};

class TruncatedLoss {
  public:
    TruncatedLoss(double threshold) : squared_thr(threshold * threshold) {}
    double loss(double r2) const {
        return std::min(r2, squared_thr);
    }
    double weight(double r2) const {
        return (r2 < squared_thr) ? 1.0 : 0.0;
    }

  private:
    const double squared_thr;
};

class HuberLoss {
  public:
    HuberLoss(double threshold) : thr(threshold) {}
    double loss(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return r2;
        } else {
            return 2.0 * thr * (r - thr);
        }
    }
    double weight(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return 1.0;
        } else {
            return thr / r;
        }
    }

  private:
    const double thr;
};
class CauchyLoss {
  public:
    CauchyLoss(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    double loss(double r2) const {
        return std::log1p(r2 * inv_sq_thr);
    }
    double weight(double r2) const {
        return std::max(std::numeric_limits<double>::min(), 1.0 / (1.0 + r2 * inv_sq_thr));
    }

  private:
    const double inv_sq_thr;
};

} // namespace pose_lib

#endif
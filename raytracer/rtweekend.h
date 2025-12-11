#ifndef RT_RTWEEKEND_H
#define RT_RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

// C++ Std Usings (kept global for compatibility)
using std::make_shared;
using std::shared_ptr;

namespace rt {

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions
inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

inline double random_double() {
    // Returns a random real in [0,1).
    return std::rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

} // namespace rt

// Common Headers (these will add to rt namespace)
#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "color.h"

#endif

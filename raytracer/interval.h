#ifndef RT_INTERVAL_H
#define RT_INTERVAL_H

namespace rt {

class interval {
  public:
    double min, max;

    interval() : min(+infinity), max(-infinity) {} // Default interval is empty

    interval(double min, double max) : min(min), max(max) {}

    double size() const {
        return max - min;
    }

    bool contains(double x) const {     // Contains x (inclusive)
        return min <= x && x <= max;
    }

    bool surrounds(double x) const {    // Strictly contains x 
        return min < x && x < max;
    }

    double clamp(double x) const {      // Clamp x to the interval, returning nearest endpoint if outside
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    static const interval empty, universe;
};

inline const interval interval::empty    = interval(+infinity, -infinity);
inline const interval interval::universe = interval(-infinity, +infinity);

} // namespace rt

#endif

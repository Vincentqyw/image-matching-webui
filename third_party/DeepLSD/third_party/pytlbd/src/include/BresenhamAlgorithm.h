/**
 * @copyright 2018 Xoan Iago Suarez Canosa. All rights reserved.
 * Constact: iago.suarez.canosa@alumnos.eth.es
 * Software developed in the PhD: Augmented Reality for Urban Environments
 */
#ifndef LINE_EXPERIMENTS_BRESENHAMALGORITHM_H
#define LINE_EXPERIMENTS_BRESENHAMALGORITHM_H

#include <vector>
#include "utils.h"

namespace eth {

/**
 * Returns the line pixels using the Bresenham Algorithm:
 * https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
 * @param x0 The X coordinate of the first endpoint
 * @param y0 The Y coordinate of the first endpoint
 * @param x1 The X coordinate of the last endpoint
 * @param y1 The Y coordinate of the last endpoint
 * @return
 */
static std::vector<Pixel> bresenham(int x0, int y0, int x1, int y1) {
  int dx, dy, p, x, y, xIncrement, yIncrement;
  std::vector<Pixel> pixels;
  dx = x1 - x0;
  dy = y1 - y0;

  // Determine the line direction
  xIncrement = dx < 0 ? -1 : +1;
  yIncrement = dy < 0 ? -1 : +1;

  x = x0;
  y = y0;
  dx = UPM_ABS(dx);
  dy = UPM_ABS(dy);
  // pixels.reserve(std::max(dx, dy));

  if (dx >= dy) {
    // Horizontal like line
    p = 2 * dy - dx;
    while (x != x1) {
      pixels.emplace_back(x, y);
      if (p >= 0) {
        y += yIncrement;
        p += 2 * dy - 2 * dx;
      } else {
        p += 2 * dy;
      }
      // Increment the axis in which we are moving
      x += xIncrement;
    }  // End of while
  } else {
    // Vertical like line
    p = 2 * dx - dy;
    while (y != y1) {
      pixels.emplace_back(x, y);
      if (p >= 0) {
        x += xIncrement;
        p += +2 * dx - 2 * dy;
      } else {
        p += 2 * dx;
      }
      // Increment the axis in which we are moving
      y += yIncrement;
    }  // End of while
  }
  pixels.emplace_back(x1, y1);
  return pixels;
}

}
#endif //LINE_EXPERIMENTS_BRESENHAMALGORITHM_H

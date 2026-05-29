#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2024 abetusk

def gilbert_d2xy(idx, w, h):
    """
    Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
    2D rectangular grids. Takes a position along the gilbert curve and returns
    its 2D (x,y) coordinate.
    """

    if w >= h:
        return gilbert_d2xy_r(idx,0, 0,0, w,0, 0,h)
    return gilbert_d2xy_r(idx,0, 0,0, 0,h, w,0)

def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)

def gilbert_d2xy_r(dst_idx, cur_idx, x,y, ax,ay, bx,by):

    w = abs(ax + ay)
    h = abs(bx + by)

    (dax, day) = (sgn(ax), sgn(ay)) # unit major direction
    (dbx, dby) = (sgn(bx), sgn(by)) # unit orthogonal direction

    dx = dax + dbx
    dy = day + dby
    di = dst_idx - cur_idx

    if h == 1: return (x + dax*di, y + day*di)
    if w == 1: return (x + dbx*di, y + dby*di)

    (ax2, ay2) = (ax//2, ay//2)
    (bx2, by2) = (bx//2, by//2)

    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)

    if 2*w > 3*h:
        if (w2 % 2) and (w > 2):
            # prefer even steps
            (ax2, ay2) = (ax2 + dax, ay2 + day)


        # long case: split in two parts only
        nxt_idx = cur_idx + abs((ax2 + ay2)*(bx + by))
        if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
            return gilbert_d2xy_r(dst_idx, cur_idx,  x, y, ax2, ay2, bx, by)
        cur_idx = nxt_idx

        return gilbert_d2xy_r(dst_idx, cur_idx, x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by)

    if (h2 % 2) and (h > 2):
        # prefer even steps
        (bx2, by2) = (bx2 + dbx, by2 + dby)

    # standard case: one step up, one long horizontal, one step down
    nxt_idx = cur_idx + abs((bx2 + by2)*(ax2 + ay2))
    if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
        return gilbert_d2xy_r(dst_idx, cur_idx, x,y, bx2,by2, ax2,ay2)
    cur_idx = nxt_idx

    nxt_idx = cur_idx + abs((ax + ay)*((bx - bx2) + (by - by2)))
    if (cur_idx <= dst_idx) and (dst_idx < nxt_idx):
        return gilbert_d2xy_r(dst_idx, cur_idx, x+bx2, y+by2, ax,ay, bx-bx2,by-by2)
    cur_idx = nxt_idx

    return gilbert_d2xy_r(dst_idx, cur_idx,
                          x+(ax-dax)+(bx2-dbx),
                          y+(ay-day)+(by2-dby),
                          -bx2, -by2,
                          -(ax-ax2), -(ay-ay2))

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('width', type=int)
    parser.add_argument('height', type=int)
    args = parser.parse_args()

    width = args.width
    height = args.height

    for idx in range(width*height):
      (x,y) = gilbert_d2xy(idx, width,height)
      print(x,y)


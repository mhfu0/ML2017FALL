#!/usr/bin/env python3

import sys, math
from PIL import Image as im

def main(argv):
    img = im.open(argv[1])
    pixels = img.load()
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            p = pixels[i,j]
            pixels[i,j] = (math.floor(p[0]/2), math.floor(p[1]/2), math.floor(p[2]/2))
    
    img.save("Q2.png")

if __name__ == "__main__":
    main(sys.argv)

#!/usr/bin/env python3

import sys

def main(argv):
    f = open(argv[1], "r")
    words = f.read().strip('\n').split(' ')
    
    wordlist = []
    wordcount = {}
    
    for w in words:
        if w not in wordlist:
            wordlist.append(w)
            wordcount[w] = 1
        else:
            wordcount[w] += 1
    
    out = open("Q1.txt", "w")
    for w in wordlist:
        outline = ' '.join([w, str(wordlist.index(w)), str(wordcount[w])]) + "\n"
        out.write(outline)

if __name__ == "__main__":
    main(sys.argv)

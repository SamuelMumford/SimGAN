#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:05:20 2021

@author: sam
"""
import os
import sys

p0 = sys.argv[1]
p1 = sys.argv[2]

print(p0)
print(p1)

path, dirs, files = next(os.walk(p0))

path1, dirs1, files1 = next(os.walk(p1))
print('initial number of files in p1: ' + str(len(files1)))

for f in files:
    if 'fake' in f:
        oldName = os.path.join(p0, f)
        newName = os.path.join(p1, f)
        string = 'cp ' + oldName + ' ' + newName
        os.popen(string)

path, dirs, files = next(os.walk(p1))
print('final number of files in p1: ' + str(len(files)))
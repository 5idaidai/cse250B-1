# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:18:32 2014

@author: axydes
"""

def writeFF(f, filename, j, a, b, A, B):
    ffunc = "def f{}(yi1, yi, x, i, n):\n".format(j)

    #print a & b differently depending on type: number, string, tag
    if isinstance(a, basestring):
        ffunc += "\ta = {2}.{1}(x, i, n, \"{0}\")\n".format(a,A.__name__,filename)
    else:
        ffunc += "\ta = {2}.{1}(x, i, n, {0})\n".format(a,A.__name__,filename)
    
    if isinstance(b, basestring):
        ffunc += "\tb = {2}.{1}(yi1, yi, \"{0}\")\n".format(b,B.__name__,filename)
    else:
        ffunc += "\tb = {2}.{1}(yi1, yi, {0})\n".format(b,B.__name__,filename)
    
    ffunc += "\treturn a*b\n\n\n"
    f.write(ffunc)


def generateFFs(templates,filename):
    """
    Input: list of FF templates
           FF template = dict('aset'={}, 'bset'={}, 'A'=funcA, 'B'=funcB)
    Output: file of feature functions       
    """
    f=open('ffs.py', 'w')
    
    f.write("import {}\n\n".format(filename))

    j = 0
    for temp in templates:
        if temp['aset']:
            for a in temp['aset']:
                if temp['bset']:
                    for b in temp['bset']:
                        writeFF(f, filename, j, a, b, temp['A'], temp['B'])
                        j += 1
                else:
                    writeFF(f, filename, j, a, 0, temp['A'], temp['B'])
                    j += 1
        else:
            if temp['bset']:
                for b in temp['bset']:
                        writeFF(f, filename, j, 0, b, temp['A'], temp['B'])
                        j += 1
            else:
                writeFF(f, filename, j, 0, 0, temp['A'], temp['B'])
                j += 1
        
    f.close()
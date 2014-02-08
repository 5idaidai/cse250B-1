# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:18:32 2014

@author: axydes
"""

aFuncs = []
bFuncs = []

def writeFF(f, filename, j, a, b, A, B):
    aFuncs.append((A,a))
    bFuncs.append((B,b))    
    
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


def writeFFAccessor(f, j):
    func = "\n\nfeatureFunc = {\n"
    
    for i in range(0,j):
        func += "\t{0} : f{0},\n".format(i)
    
    func += "}\n"
    
    f.write(func)
    

def writeAAccessor(f, filename):
    func = "\n\naFunc = {\n"
    
    for i in range(0,len(aFuncs)):
        func += "\t{0} : Func(func={1}.{2},val=".format(i,filename,aFuncs[i][0].__name__)
        if isinstance(aFuncs[i][1], basestring):
            func += "\"{}\"),\n".format(aFuncs[i][1])
        else:
            func += "{}),\n".format(aFuncs[i][1])
            
    func += "}\n"
    
    f.write(func)
    
    
def writeBAccessor(f, filename):
    func = "\n\nbFunc = {\n"
    
    for i in range(0,len(bFuncs)):
        func += "\t{0} : Func(func={1}.{2},val=".format(i,filename,bFuncs[i][0].__name__)
        if isinstance(bFuncs[i][1], basestring):
            func += "\"{}\"),\n".format(bFuncs[i][1])
        else:
            func += "{}),\n".format(bFuncs[i][1])
    
    func += "}\n"
    
    f.write(func)


def generateFFs(templates,filename):
    """
    Input: list of FF templates
           FF template = dict('aset'={}, 'bset'={}, 'A'=funcA, 'B'=funcB)
    Output: file of feature functions       
    """
    f=open('ffs.py', 'w')
    
    f.write("import {}\n\n".format(filename))
    f.write("from collections import namedtuple\n")
    f.write("Func = namedtuple(\"Func\", [\"func\", \"val\"])\n\n")

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
                
    writeAAccessor(f, filename)
    writeBAccessor(f, filename)
    writeFFAccessor(f, j)
                
    f.write("\nnumJ={}\n".format(j))
        
    f.close()
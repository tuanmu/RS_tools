#!/usr/bin/env python
#-------------------------------------------------------------------------------
# Name:        Aggregation_GLCM.py
# Purpose:     Aggregating pixel values based on second-order (GLCM) texture measures
#
# Author:      Mao-Ning Tuanmu
#
# Created:     05/13/2013
# Modified:    02/11/2015 - Replace for loops with array functions in metric calculations
# Copyright:   (c) tuanmu 2013
# Licence:     <your licence>
#
# -------------------------------------------------------------------------------
# Usage:       Aggregation_GLCM.py [-m "metric_name [metric_name...]"] [-o output_folder]
#                             [-rxy ref_x ref_y] [-pn n_pixel_x n_pixel_y]
#                             [-ps pixelsize_x pixelsize_y] [-ref ref_image]
#                             [-l level] [-mm maxmin] [-f output_format] [-no noDataValue]
#                             [-qa] input_image
#
# Arguments:   -m       metrics to calculate
#                           available metrics: ASM,CON,COR,DIS,ENT,GLCMMAX,GLCMVAR,
#                                           GLCMSD,GLCMMEAN,HOM
#              -o       output folder (full directory)
#                           one image for each metric will be created in the output folder
#                           default: the same folder where the input image locates
#              -rxy     reference point
#                           x and y coordinates used as a reference point
#                           default: x and y in the upper left corner of the input image
#              -pn      times larger than the pixel size of the input image
#                           default: 3 3
#              -ps      pixel size in the unit of the input image
#                           override the argument -pn
#              -ref     reference image (full directory)
#                           reference point and pixel size will be obtained from
#                           the reference image and override the arguments -rxy -pn -ps
#              -l       GLCM level
#                           default: 64
#              -mm      Maximum and minimum pixel values
#                           default: None
#              -f       image format
#                           default: GTiff
#              -no      NoData value
#                           default: -9999
#              -qa      output quality layer (# of non-Null pixels)
#              -scale   rescale the input image
#
# -------------------------------------------------------------------------------



# import required modules
import scipy as sp
from osgeo import gdal
import math
import sys

# define some functions
def reScale(Array, MaxMin=None, level=64, NoData=-9999):
    '''Rescale pixel values

    MaxMin should be a list containing max and min (max,min)
    it will be calculated from the inputed array if it is not provided'''


    if isinstance(Array, sp.ma.MaskedArray):
        Array = Array.astype(float)
    else:
        Array = sp.ma.masked_values(Array, NoData).astype(float)

    if MaxMin == None:
        Max = Array.max()
        Min = Array.min()
        Range = Max-Min
    else:
        Max = MaxMin[0]
        Min = MaxMin[1]
        Range = Max-Min

    Array = sp.where(Array<Min, Min, Array)
    Array = sp.where(Array>Max, Max, Array)
    newArray = ((Array - Min)/Range*(level-1)).round()

    return newArray

def GLCM(Array, level, orient=1, NoData=-9999):
    '''Generates a GLCM from an array with all four directions

    '''

    if type(Array) is not sp.ma.core.MaskedArray:
        Array = sp.ma.masked_values(Array, NoData)

    if orient == 1:
        newArray = sp.zeros([level, level])
        for r in range(Array.shape[0]):
            for c in range(Array.shape[1]-1):
                if not Array.mask[r,c] and not Array.mask[r,c+1]:
                    newArray[Array[r,c], Array[r,c+1]] += 1
                    newArray[Array[r,c+1], Array[r,c]] += 1
        if newArray.sum() != 0:
            newArray = newArray/newArray.sum()

    if orient == 2:
        newArray = sp.zeros([level, level])
        for r in range(Array.shape[0]-1):
            for c in range(Array.shape[1]):
                if not Array.mask[r,c] and not Array.mask[r+1,c]:
                    newArray[Array[r,c], Array[r+1,c]] += 1
                    newArray[Array[r+1,c], Array[r,c]] += 1
        if newArray.sum() != 0:
            newArray = newArray/newArray.sum()

    if orient == 3:
        newArray = sp.zeros([level, level])
        for r in range(Array.shape[0]-1):
            for c in range(Array.shape[1]-1):
                if not Array.mask[r,c] and not Array.mask[r+1,c+1]:
                    newArray[Array[r,c], Array[r+1,c+1]] += 1
                    newArray[Array[r+1,c+1], Array[r,c]] += 1
        if newArray.sum() != 0:
            newArray = newArray/newArray.sum()

    if orient == 4:
        newArray = sp.zeros([level, level])
        for r in range(Array.shape[0]-1):
            for c in range(1, Array.shape[1]):
                if not Array.mask[r,c] and not Array.mask[r+1,c-1]:
                    newArray[Array[r,c], Array[r+1,c-1]] += 1
                    newArray[Array[r+1,c-1], Array[r,c]] += 1
        if newArray.sum() != 0:
            newArray = newArray/newArray.sum()

    return newArray



def CON(glcm):
    '''Calculates Contrast from a GLCM'''

#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]*((r-c)**2)

    ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
    ra = sp.transpose(ca)
    rc = abs(ca-ra)
    value = (glcm*(rc*rc)).sum()

    return value


def DIS(glcm):
    '''Calculates Dissimilarity from a GLCM'''

#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]*abs(r-c)

    ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
    ra = sp.transpose(ca)
    rc = abs(ca-ra)
    value = (glcm*rc).sum()

    return value


def HOM(glcm):
    '''Calculates Homogeneity from a GLCM'''

#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]/((r-c)**2 + 1)

    ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
    ra = sp.transpose(ca)
    rc = abs(ca-ra)
    value = (glcm/((rc*rc)+1)).sum()

    return value



def ASM(glcm):
    '''Calculates Angular Second Moment from a GLCM'''

#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]**2

    value = sp.square(glcm).sum()
    return value


def GLCMMAX(glcm):
    '''Calculates Maximum Probability from a GLCM'''

    return glcm.max()


def ENT(glcm):
    '''Calculates Entropy from a GLCM'''

#    import math
#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]*(math.log(glcm[r,c])*(-1))

    glcm = sp.ma.masked_equal(glcm, 0)
    value = -1*((glcm*(sp.ma.log(glcm))).sum())

    return value


def GLCMMEAN(glcm):
    '''Calculates GLCM Mean from a GLCM'''

#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]*r

    ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
    value = (glcm*ca).sum()

    return value


def GLCMVAR(glcm):
    '''Calculates GLCM Variance from a GLCM'''

    m = GLCMMEAN(glcm)
#    value = 0
#    for r in range(glcm.shape[0]):
#        for c in range(glcm.shape[1]):
#            if glcm[r,c] != 0:
#                value += glcm[r,c]*(r-m)**2

    ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
    value = (glcm*((ca-m)**2)).sum()

    return value


def GLCMSD(glcm):
    '''Calculates GLCM SD from a GLCM'''

    return GLCMVAR(glcm)**0.5


def COR(glcm):
    '''Calculates Correlation from a GLCM'''

    m = GLCMMEAN(glcm)
    v = GLCMVAR(glcm)
#    value = 0
    if v != 0:
#        for r in range(glcm.shape[0]):
#            for c in range(glcm.shape[1]):
#                if glcm[r,c] != 0:
#                    value += glcm[r,c] * (((r-m)*(c-m))/v)

        ca = sp.array([range(1,glcm.shape[1]+1)]*glcm.shape[0])
        ra = sp.transpose(ca)
        value = (glcm*((ra-m)*(ca-m))/v).sum()

    else:
        value = 1

    return value



def calcMetric(glcm, metric):
    '''Calculates multiple texture measures based on the GLCM calculated with all directions

        glcm is an array generated by using GLCM2

        metric is a list of the name of texture measures
        it could be a subset of ['CON','DIS','HOM','ASM',
        'GLCMMAX','ENT','GLCMMEAN','GLCMVAR','GLCMSD','COR']

        returns a dictionary of metrics'''

    metricDic = {}

    for m in metric:
        metricDic[m] = eval(m + "(glcm)")

    return metricDic


# =============================================================================
def Usage():
    print('Usage: aggregation_GLCM.py [-m "metric_name [metric_name...]"] [-o output_folder]')
    print('                           [-rxy ref_x ref_y] [-pn n_pixel_x n_pixel_y]')
    print('                           [-ps pixelsize_x pixelsize_y] [-ref ref_image]')
    print('                           [-l level] [-mm max_min] [-f output_format]')
    print('                           [-no noDataValue] [-qa] input_image')
    print('                           [--help-general]')
    print('')

# =============================================================================


# set default values
metric = ['CON','DIS','HOM','ASM','GLCMMAX','ENT','GLCMMEAN','GLCMVAR','GLCMSD','COR']
outPath = None
refX = None
refY = None
winX = 3
winY = 3
pSizeX = None
pSizeY = None
refImg = None
level = 64
MaxMin = None
fm = 'GTiff'
noData = None
inFile = None
outputQA = False
scale = False

# obtain arguments
arg = sys.argv
i = 1
while i < len(arg):
    if arg[i] == '-m':
        i = i + 1
        if arg[i] != 'all':
            metric = arg[i].split()
    elif arg[i] == '-o':
        i = i + 1
        outPath = arg[i]

    elif arg[i] == '-rxy':
        refX = float(arg[i+1])
        refY = float(arg[i+2])
        i = i + 2
    elif arg[i] == '-pn':
        winX = int(arg[i+1])
        winY = int(arg[i+2])
        i = i + 2
    elif arg[1] == '-ps':
        pSizeX = float(arg[i+1])
        pSizeY = -1 * abs(float(arg[i+2]))
        i = i + 2
    elif arg[i] == '-ref':
        i = i + 1
        refImg = arg[i]
    elif arg[i] == '-l':
        i = i + 1
        level = int(arg[i])
    elif arg[i] == '-mm':
        MaxMin = [float(arg[i+1]), float(arg[i+2])]
        i = i + 2
    elif arg[i] == '-f':
        i = i + 1
        fm = arg[i]
    elif arg[i] == '-no':
        i = i + 1
        noData = float(arg[i])
    elif arg[i] == '-qa':
        outputQA = True
    elif arg[i] == '-scale':
        scale = True
    elif arg[i][:1] == '-':
        print('Unrecognised command option: %s' % arg)
        Usage()
        sys.exit( 1 )
    else:
        inFile = arg[i]
    i = i + 1

# set initial values
if inFile == None:
    print('No input file provided.')
    Usage()
    sys.exit( 1 )
else:
    g = gdal.Open(inFile)
    proj = g.GetProjection()
    geo = g.GetGeoTransform()
    if noData == None:
        noData = g.GetRasterBand(1).GetNoDataValue()
    arr = sp.ma.masked_values(g.ReadAsArray(), noData)
    if scale:
        arr = reScale(arr, MaxMin, level, noData)
    nX = g.RasterXSize
    nY = g.RasterYSize
    g = None
    if inFile.rfind('/') < 0:
        filename = inFile[inFile.rfind('\\')+1:]
    else:
        filename = inFile[inFile.rfind('/')+1:]

if outPath == None:
    outPath = inFile[:inFile.rfind('/')+1]
    if outPath == '':
        outPath = inFile[:inFile.rfind('\\')+1]
elif '/' in outPath:
    outPath = outPath + '/'
elif '\\' in outPath:
    outPath = outPath + '\\'

if refImg != None:
    gref = gdal.Open(refImg)
    refgeo = gref.GetGeoTransform()
    refX = refgeo[0]
    refY = refgeo[3]
    pSizeX = refgeo[1]
    pSizeY = refgeo[5]
    gref = None
elif refX == None or refY == None:
    refX = geo[0]
    refY = geo[3]
    if pSizeX == None or pSizeY == None:
        pSizeX = geo[1] * winX
        pSizeY = geo[5] * winY
elif pSizeX == None or pSizeY == None:
    pSizeX = geo[1] * winX
    pSizeY = geo[5] * winY


# define the extent of the grid
LX = (math.ceil((geo[0]-refX)/pSizeX)) * pSizeX + refX
NX = int(((math.floor(((geo[0]+geo[1]*nX)-refX)/pSizeX)) - (math.ceil((geo[0]-refX)/pSizeX))))
UY = (math.ceil((geo[3]-refY)/pSizeY)) * pSizeY + refY
NY = int(((math.floor(((geo[3]+geo[5]*nY)-refY)/pSizeY)) - (math.ceil((geo[3]-refY)/pSizeY))))

# create new arrays
arrayDic = {}
for m in metric:
    newArray = sp.empty((NY,NX))
    newArray.fill(noData)
    arrayDic[m] = newArray
if outputQA:
    countArray = sp.zeros((NY,NX))

# calculate texture metrics
#toolbar_width = 40
#sys.stdout.write("[%s]" % (" " * toolbar_width))
#sys.stdout.flush()
#sys.stdout.write("\b" * (toolbar_width+1))

for R in range(NY):
    uy = UY + R*pSizeY
    ly = uy + pSizeY
    r_start = int(round((uy-geo[3])/geo[5]))
    r_end = int(round((ly-geo[3])/geo[5]))

    for C in range(NX):
        lx = LX + C*pSizeX
        rx = lx + pSizeX
        c_start = int(round((lx-geo[0])/geo[1]))
        c_end = int(round((rx-geo[0])/geo[1]))

        win = arr[r_start:r_end,c_start:c_end]
        if outputQA:
            countArray[R,C] = float(win.count())/((r_end-r_start)*(c_end-c_start))
        if win.count() != 0:
            glcm1 = GLCM(win, level, 1, noData)
            glcm2 = GLCM(win, level, 2, noData)
            glcm3 = GLCM(win, level, 3, noData)
            glcm4 = GLCM(win, level, 4, noData)  # call the function
            valueDic1 = calcMetric(glcm1, metric)
            valueDic2 = calcMetric(glcm2, metric)
            valueDic3 = calcMetric(glcm3, metric)
            valueDic4 = calcMetric(glcm4, metric)
            for m in metric:
                arrayDic[m][R,C] = (valueDic1[m] + valueDic2[m] + valueDic3[m] + valueDic4[m])/4.0

#    if (R+1) % (NY/toolbar_width) == 0:
#        sys.stdout.write('-')
#        sys.stdout.flush()
#sys.stdout.write('\n')


# set new geo
newGeo = (LX, pSizeX, 0.0, UY, 0.0, pSizeY)

# export arrays
driver = gdal.GetDriverByName(fm)

for m in metric:
    outFile = driver.Create(outPath + m + '_' + filename, NX, NY, 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
    outFile.SetGeoTransform( newGeo ) # set the datum
    outFile.SetProjection( proj )  # set the projection
    outFile.GetRasterBand(1).WriteArray(arrayDic[m])  # write numpy array band1 as the first band of the multiTiff - this is the blue band
    stat = outFile.GetRasterBand(1).GetStatistics(1,1)  # get the band statistics (min, max, mean, standard deviation)
    outFile.GetRasterBand(1).SetStatistics(stat[0], stat[1], stat[2], stat[3])  # set the stats we just got to the band
    outFile.GetRasterBand(1).SetNoDataValue(noData)
    outFile = None

if outputQA:
    outFile = driver.Create(outPath + 'pValid_' + filename, NX, NY, 1, gdal.GDT_Float32, ['COMPRESS=LZW'])
    outFile.SetGeoTransform( newGeo ) # set the datum
    outFile.SetProjection( proj )  # set the projection
    outFile.GetRasterBand(1).WriteArray(countArray)  # write numpy array band1 as the first band of the multiTiff - this is the blue band
    stat = outFile.GetRasterBand(1).GetStatistics(1,1)  # get the band statistics (min, max, mean, standard deviation)
    outFile.GetRasterBand(1).SetStatistics(stat[0], stat[1], stat[2], stat[3])  # set the stats we just got to the band
    outFile.GetRasterBand(1).SetNoDataValue(noData)
    outFile = None



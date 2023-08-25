#!/bin/python

from os.path import isdir
from re import L
from typing import final
import xml.etree.ElementTree as ET
import sys,os
import parse
from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np
import math
from matplotlib import pyplot as plt

from sklearn import linear_model


if len(sys.argv) < 5:
    print(sys.argv[0] + " TRIPLETPATH ORI1 ORI2 OUT [test|random]")
    exit(0)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

TRIP_NAME = sys.argv[1]
ORI1_NAME = sys.argv[2]
ORI2_NAME = sys.argv[3]
ORIOUT_NAME = sys.argv[4]

testBascule = False
testRandom = False

if len(sys.argv) == 6:
    if sys.argv[5] == 'test':
        testBascule = True
    if sys.argv[5] == 'random':
        testRandom = True


def gs(X):
    Q, R = np.linalg.qr(X)
    return Q

class Image():
    def __init__(self, xml, name, pos, rot):
        self.name = name
        self.pos = pos
        #self.rot = gs(rot)
        self.rot = rot
        self.xml = xml

    def __str__(self):
        return "(" + self.name + ")" + str(self.pos) + "|" + str(self.rot)

    def inv(self):
        return Image(self.xml, self.name, -1 * (self.rot@self.pos), self.rot.transpose())

class Triplet():
    def __init__(self, names, pos, rot):
        self.names = names
        self.pos = pos
        self.rot = rot

    def __str__(self):
        r = ""
        for i in [0, 1, 2]:
            r += self.names[i]
            r += "\n"
            r += str(self.pos[i])
            r += "\n"
            r += str(self.rot[i])
            r += "\n"
            i += 1
        return r

"""
Load an array position from xml MicMac structure
"""
def xml_load_pos(e):
    pos = [0.] * 3
    i = 0
    for e in e.find("Centre").text.split(" "):
        pos[i] = float(e)
        i += 1
    return np.array(pos)

def xml_write_pos(e, pos):
    e.find("Centre").text = str(pos[0]) + " " + str(pos[1]) + " " + str(pos[2])
    return

"""
Load a rotation matrix from xml MicMac structure
"""
def xml_load_rot(e):
    #rot = [0.] * 9
    rot = []
    #j = 0
    for l in ["L1", "L2", "L3"]:
        #i = 0
        line = []
        for c in e.find(l).text.split(" "):
            vc = np.longdouble(c)
            line.append(vc)
            #rot[j * 3 + i] = float(c)
            #i += 1
        rot.append(line)
        #j += 1
    return np.array(rot)

def xml_write_rot(e, rot):
    i = 0
    for l in ["L1", "L2", "L3"]:
        e.find(l).text = str(rot[i, 0]) + " " + str(rot[i, 1]) + " " + str(rot[i, 2])
        i += 1
    return np.array(rot)

"""
Load the MicMac View Orientation XML folder.
"""
def load_images(path, offset_rot = np.identity(3), offset_tr = [0,0,0],
                offset_lamda = 1.):
    images = {}
    with os.scandir(path + '/') as entries:
        for entry in entries:
            if entry.name.startswith("Orientation-"):
                tree = ET.parse(path + '/' + entry.name)
                root = tree.getroot()
                if "ExportAPERO" == root.tag: #Handle when everything inside
                    root = root.find("OrientationConique")

                c = root.find('Externe')
                pos = xml_load_pos(c)
                rot = xml_load_rot(c.find("ParamRotation").find('CodageMatr'))
                name = parse.parse("Orientation-{}.xml", entry.name)[0]

                images[name] = Image(tree, name, offset_rot @ (offset_lamda *
                    (pos+offset_tr)),
                                     offset_rot @ rot)
    return images

def save_images(path, images):
    if not os.path.isdir(path):
        os.mkdir(path)
    ## TODO copy calibration file
    for i in images:
        file = path + '/' + "Orientation-{}.xml".format(i.name)
        root = i.xml.getroot()
        if "ExportAPERO" == root.tag: #Handle when everything inside
            root = root.find("OrientationConique")

        c = root.find('Externe')
        xml_write_pos(c, i.pos)
        xml_write_rot(c.find("ParamRotation").find('CodageMatr'), i.rot)
        i.xml.write(file)

    return

"""
Load the MicMac Triplet XML folder.

First load the triplet list file.
Then load all the triplet files to get local orientations
"""
def load_triplet_list(path):
    triplets_list = []
    with os.scandir(path + '/') as entries:
        for entry in entries:
            if entry.name.startswith("ListeTriplets") and \
                entry.name.endswith(".xml"):
                root = ET.parse(path + '/' + entry.name).getroot()
                for c in root.findall('Triplets'):
                    names = [c.find("Name1").text, c.find("Name2").text,
                             c.find("Name3").text]
                    pos = []
                    pos.append(np.zeros(3))
                    rot = []
                    rot.append(np.identity(3))
                    r = ET.parse(path + "/" + names[0] + "/"\
                                 + names[1] + "/Triplet-OriOpt-" + names[2] + ".xml").getroot()
                    pos.append(xml_load_pos(r.find("Ori2On1")))
                    rot.append(xml_load_rot(r.find("Ori2On1").find("Ori")))

                    pos.append(xml_load_pos(r.find("Ori3On1")))
                    rot.append(xml_load_rot(r.find("Ori3On1").find("Ori")))

                    triplets_list.append(Triplet(names, pos, rot))
    return triplets_list

"""
Check if the images in the two block are unique
"""
def check_unique(images1, images2):
    for i in images2:
        if i in images1:
            return False
    return True

"""
Takes an array of rotation and average all of them.
Then it compute the closest orthogonal rotation of this average.
"""
def mean_rotation(rots):

    allrot = np.zeros((3,3))
    for r in rots:
        allrot = allrot + r
    allrot = allrot * (1.0 / np.longdouble(len(rots)))

    u, s, vh = np.linalg.svd(allrot.astype(float))
    ns = np.identity(3)
    for i in range(3):
        if s[i] > 0.:
            ns[i,i] = 1.
        else:
            ns[i,i] = -1.
    return u @ ns @ vh

def ransac(data,model,n,k,t,d,debug=False,return_all=False):
    """fit model parameters to data using the RANSAC algorithm

This implementation written from pseudocode found at
http://en.wikipedia.org/w/index.php?title=RANSAC&oldid=116358182

{{{
Given:
    data - a set of observed data points
    model - a model that can be fitted to data points
    n - the minimum number of data values required to fit the model
    k - the maximum number of iterations allowed in the algorithm
    t - a threshold value for determining when a data point fits a model
    d - the number of close data values required to assert that a model fits well to data
Return:
    bestfit - model parameters which best fit the data (or nil if no good model is found)
iterations = 0
bestfit = nil
besterr = something really large
while iterations < k {
    maybeinliers = n randomly selected values from data
    maybemodel = model parameters fitted to maybeinliers
    alsoinliers = empty set
    for every point in data not in maybeinliers {
        if point fits maybemodel with an error smaller than t
             add point to alsoinliers
    }
    if the number of elements in alsoinliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        bettermodel = model parameters fitted to all points in maybeinliers and alsoinliers
        thiserr = a measure of how well model fits these points
        if thiserr < besterr {
            bestfit = bettermodel
            besterr = thiserr
        }
    }
    increment iterations
}
return bestfit
}}}
"""
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = model.random_partition(n,data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs, :]
        maybemodel = model.fit(maybeinliers)
        test_err = model.get_error( test_points, maybemodel)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if debug:
            print('test_err.min()',test_err.min())
            print('test_err.max()',test_err.max())
            print('numpy.mean(test_err)',np.mean(test_err))
        if len(alsoinliers) > d:
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error( betterdata, bettermodel)
            thiserr = np.mean( better_errs )
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate( (maybe_idxs, also_idxs) )
        iterations+=1
    if bestfit is None:
        return [], {}, False
    if return_all:
        return bestfit, {'inliers':best_inlier_idxs}, True
    else:
        return bestfit, [], True

class MeanRotationModel:
    def __init__(self, images,debug=False):
        self.images = images
        self.debug = debug

    def random_partition(self, n, n_data):
        """return n random rows of data (and also the other len(data)-n rows)"""
        all_idxs = np.arange( n_data )
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

    def rotation(self, t):
        images = self.images
        names = [t.names[t.m[0]], t.names[t.m[1]], t.names[t.m[2]]]
        r_t1_b1 = images[names[0]].rot @ t.rot[t.m[0]].transpose()
        r3_b1 = r_t1_b1 @ t.rot[t.m[2]]
        r_b2_b1 = images[names[2]].rot@r3_b1.transpose()

        r_t1_b2 = images[names[2]].rot @ t.rot[t.m[2]].transpose()
        r1_b2 = r_t1_b2 @ t.rot[t.m[0]]
        r_b1_b2_1 = images[names[0]].rot @ r1_b2.transpose()

        #If V3 is in block 1 we invert rotation
        if t.b[t.m[2]] == 2:
            rot1 = r_b2_b1
            rot2 = r_b1_b2_1.transpose()
        else:
            rot1 = r_b2_b1.transpose()
            rot2 = r_b1_b2_1
        return rot1, rot2

    def fit(self, ts):
        rot = []
        for t in ts[:, 0]:
            rot1, rot2 = self.rotation(t)
            rot.append(rot1)
            rot.append(rot2)
        return mean_rotation(rot)


    def get_error(self, data, rotBascule):
        #ref = R.from_matrix(rotBascule).as_euler('xyz')
        err_per_point = []
        for t in data[:,0]:
            rot1, rot2 = self.rotation(t)
            mean = mean_rotation([rot1, rot2])
            #rot = R.from_matrix(mean).as_euler('xyz')
            err = math.acos(((mean @ rotBascule.transpose()).trace()-1.)/2.)
            err_per_point.append(err)
            #err_per_point.append(np.linalg.norm(ref-rot))
        return np.array(err_per_point)



class LinearLeastSquaresModel:
    """linear system solved using linear least squares

    This class serves as an example that fulfills the model interface
    needed by the ransac() function.

    """
    def __init__(self,bascule_idx, input_columns,output_columns,debug=False):
        self.bascule_idx = bascule_idx
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def random_partition(self, n, n_data):
        """return n random rows of data (and also the other len(data)-n rows)"""
        number_eq = 9
        n_data //= number_eq
        n //= number_eq
        all_idxs = np.arange( n_data )
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]

        f1 = []
        for i in idxs1:
            for j in range(number_eq):
                f1.append(i*number_eq + j)
        f2 = []
        for i in idxs2:
            for j in range(number_eq):
                f2.append(i*number_eq + j)

        return np.array(f1).astype(int), np.array(f2).astype(int)

    def fit(self, data):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        x,resids,rank,s = scipy.linalg.lstsq(A,B, cond=0.1, overwrite_a=True,
                                             overwrite_b=True, lapack_driver="gelsd")
        return x

    def get_error( self, data, model):
        A = np.vstack([data[:,i] for i in self.input_columns]).T
        B = np.vstack([data[:,i] for i in self.output_columns]).T
        B_fit = np.dot(A,model)
        err_per_point = np.sum((B-B_fit)**2,axis=1) # sum squared error per row
        return err_per_point



"""
Takes a list of images and compute the rotation going from block2 to block1
"""
def compute_rotation(images, triplets):

    finalrot = np.identity(3)
    good = False

    if len(triplets) > 10:
        debug = False
        model = MeanRotationModel(images, debug=debug)

        p = 0.9
        s = 2
        e = 0.9

        n = math.log(1.-p)/math.log(1.-math.pow(1.-e,s))
        print("N", n)
        n = 10
        data = np.array(triplets).reshape(len(triplets), 1)

        # run RANSAC algorithm
        ransac_fit, ransac_data, good = ransac(data, model,
                                         3, n, 0.0001, 6, # misc. parameters
                                         debug=debug, return_all=True)

        print("fit", ransac_fit)
        print("data", ransac_data)
        if good:
            finalrot = ransac_fit.transpose()

    if not good:
        rot = []
        for t in triplets:
            #Rotate triplet on B1
            names = [t.names[t.m[0]], t.names[t.m[1]], t.names[t.m[2]]]

            r_t1_b1 = images[names[0]].rot @ t.rot[t.m[0]].transpose()
            r3_b1 = r_t1_b1 @ t.rot[t.m[2]]
            r_b2_b1 = images[names[2]].rot@r3_b1.transpose()

            r_t1_b2 = images[names[2]].rot @ t.rot[t.m[2]].transpose()
            r1_b2 = r_t1_b2 @ t.rot[t.m[0]]
            r_b1_b2_1 = images[names[0]].rot @ r1_b2.transpose()

            #If V3 is in block 1 we invert rotation
            if t.b[t.m[2]] == 2:
                rot.append(r_b2_b1)
                rot.append(r_b1_b2_1.transpose())
            else:
                rot.append(r_b2_b1.transpose())
                rot.append(r_b1_b2_1)

        finalrot = mean_rotation(rot).transpose()

    print(finalrot)

    return finalrot

def computeall_tr_u(rot, images1, images2, images, triplets):
    if len(triplets) < 2:
        #Need 2 triplet to work ?
        return 0,0,False

    for n,i in images2.items():
        images[n].pos = rot @ i.pos

    #First rotate triplet from block1
    for t in triplets:
        view0 = t.m[0]
        if t.b[view0] == 2:
            view0 = t.m[2]
        r = images[t.names[view0]].rot @ t.rot[view0].transpose()
        for i in range(3):
            t.rot[i] = r @ t.rot[i]
            t.pos[i] = r @ t.pos[i]

    t_normal = []
    t_inverted = []
    for t in triplets:
        if t.b[t.m[2]] == 2:
            t_normal.append(t)
        else:
            t_inverted.append(t)

    print("Normal: ", len(t_normal), ' Inverted: ', len(t_inverted))

    n_x = 4 * len(triplets) + 4
    n_y = 9 * len(triplets)
    a = np.zeros((n_y, n_x), dtype=float)
    b = np.zeros((n_y, 1), dtype=float)

    bascule_idx = []

    n_t = 0
    for t in t_normal:
        B = np.array([images[t.names[t.m[0]]].pos[0],
        images[t.names[t.m[0]]].pos[1],
        images[t.names[t.m[0]]].pos[2],
        images[t.names[t.m[1]]].pos[0],
        images[t.names[t.m[1]]].pos[1],
        images[t.names[t.m[1]]].pos[2],
        0.,
        0.,
        0.])
        r = n_t * 9
        b[r:r+B.shape[0], 0] = B

        A = np.array([
            [t.pos[t.m[0]][0],1.,0.,0.],
            [t.pos[t.m[0]][1],0.,1.,0.],
            [t.pos[t.m[0]][2],0.,0.,1.],
            [t.pos[t.m[1]][0],1.,0.,0.],
            [t.pos[t.m[1]][1],0.,1.,0.],
            [t.pos[t.m[1]][2],0.,0.,1.],
            [-t.pos[t.m[2]][0],-1.,0.,0.],
            [-t.pos[t.m[2]][1],0.,-1.,0.],
            [-t.pos[t.m[2]][2],0.,0.,-1.]
        ])
        r,c = (n_t * 9, n_t * 4)
        a[r:r+A.shape[0], c:c+A.shape[1]] = A

        A1 = np.array([
            [images[t.names[t.m[2]]].pos[0],1.,0.,0.],
            [images[t.names[t.m[2]]].pos[1],0.,1.,0.],
            [images[t.names[t.m[2]]].pos[2],0.,0.,1.],
        ])
        r,c = (n_t * 9 + 6, a.shape[1] - 4)
        a[r:r+A1.shape[0], c:c+A1.shape[1]] = A1

        bascule_idx.append(n_t * 9 + 6)
        bascule_idx.append(n_t * 9 + 7)
        bascule_idx.append(n_t * 9 + 8)

        n_t += 1

    for t in t_inverted:
        B = np.array([images[t.names[t.m[2]]].pos[0],
        images[t.names[t.m[2]]].pos[1],
        images[t.names[t.m[2]]].pos[2],
        0.,
        0.,
        0.,
        0.,
        0.,
        0.])
        r = n_t * 9
        b[r:r+B.shape[0], 0] = B

        A = np.array([
            [t.pos[t.m[2]][0],1.,0.,0.],
            [t.pos[t.m[2]][1],0.,1.,0.],
            [t.pos[t.m[2]][2],0.,0.,1.],
            [-t.pos[t.m[0]][0],-1.,0.,0.],
            [-t.pos[t.m[0]][1],0.,-1.,0.],
            [-t.pos[t.m[0]][2],0.,0.,-1.],
            [-t.pos[t.m[1]][0],-1.,0.,0.],
            [-t.pos[t.m[1]][1],0.,-1.,0.],
            [-t.pos[t.m[1]][2],0.,0.,-1.]
        ])
        r,c = (n_t * 9, n_t * 4)
        a[r:r+A.shape[0], c:c+A.shape[1]] = A

        A1 = np.array([
            [images[t.names[t.m[0]]].pos[0],1.,0.,0.],
            [images[t.names[t.m[0]]].pos[1],0.,1.,0.],
            [images[t.names[t.m[0]]].pos[2],0.,0.,1.],
            [images[t.names[t.m[1]]].pos[0],1.,0.,0.],
            [images[t.names[t.m[1]]].pos[1],0.,1.,0.],
            [images[t.names[t.m[1]]].pos[2],0.,0.,1.],
        ])
        r,c = (n_t * 9 + 3, a.shape[1] - 4)
        a[r:r+A1.shape[0], c:c+A1.shape[1]] = A1

        bascule_idx.append(n_t * 9 + 3)
        bascule_idx.append(n_t * 9 + 4)
        bascule_idx.append(n_t * 9 + 5)
        bascule_idx.append(n_t * 9 + 6)
        bascule_idx.append(n_t * 9 + 7)
        bascule_idx.append(n_t * 9 + 8)

        n_t += 1

    good = False
    r_tr = [0,0,0]
    r_u = -1

    #if len(triplets) > 10:
    if len(triplets) > 4 and len(triplets) < 30:
        n_inputs = len(triplets) * 4 + 4
        n_outputs = 1
        all_data = np.hstack( (a,b) )
        input_columns = range(n_inputs) # the first columns of the array
        output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
        debug = False
        model = LinearLeastSquaresModel(bascule_idx, input_columns,output_columns,debug=debug)

        p = 0.9
        s = 3
        e = 0.8

        n = math.log(1.-p)/math.log(1.-math.pow(1.-e,s))
        print("N", n)
        n = 20

        # run RANSAC algorithm
        ransac_fit, ransac_data, good = ransac(all_data, model,
                                         2, n, 0.001, 3, # misc. parameters
                                         debug=debug, return_all=True)

        print("fit", ransac_fit)
        print("data", ransac_data)
        if good:
            x = ransac_fit
            e = len(x)
            r_tr = np.array([x[e-3], x[e-2], x[e-1]])
            r_u = x[e-4]

    if not good or r_u < 0:
        x, res, rank, s = np.linalg.lstsq(a.astype(float), b.astype(float),
                                      rcond=None)

        np.set_printoptions(suppress=True)
        #print("res", res)
        #print("rank", rank)
        #print("s", s)
        #print("x", x)

        e = len(x)
        r_tr = np.array([x[e-3], x[e-2], x[e-1]])
        r_u = x[e-4]

    return r_tr, r_u, True

def compute_bascule(images, images1, images2, triplets):
    rot = compute_rotation(images, triplets)
    np.set_printoptions(suppress=True)
    #print("Bascule", rot)
    #print('EulerRot', R.from_matrix(rot).as_euler('XYZ', degrees=True))

    tr,u,err = computeall_tr_u(rot, images1, images2, images, triplets)

    return rot,tr,u,err


def main():
    BasculeRot = np.array([[0.9848077, -0.1736482,  0.0000000],
        [-0.0868241, -0.4924039, -0.8660254],
        [0.1503837,  0.8528686, -0.5000000]])
    BasculeRot = R.from_euler('XYZ', [0, 10, 0], degrees=True).as_matrix()
    # To disable rotation bascule
    #BasculeRot = np.identity(3)
    BasculeLambda = 10.
    BasculeTr = np.array([10, 0, 0])

    #Load the two block folders
    #images1 = load_images(ORI1_NAME, BasculeRot, BasculeTr, BasculeLambda)
    images1 = load_images(ORI1_NAME)

    images2_ori = load_images(ORI2_NAME)
    if testBascule:
        images2 = load_images(ORI2_NAME, BasculeRot, BasculeTr, BasculeLambda)
    else:
        images2 = images2_ori

    simages1 = set()
    for n,i in images1.items():
        simages1.add(i.name)
    simages2 = set()
    for n,i in images2.items():
        simages2.add(i.name)

    if not check_unique(simages1, simages2):
        print("Same image in both block")
        exit(1)

    triplets_list_all = load_triplet_list(TRIP_NAME)
    triplets_list = []

    # Check for each triplet if one side is in one block and the two other in the
    # other block
    for t in triplets_list_all:
        in1 = 0
        in2 = 0
        mask = [0]*3
        for i in [0, 1, 2]:
            if t.names[i] in simages1:
                in1 += 1
                mask[i] = 1
            if t.names[i] in simages2:
                in2 += 1
                mask[i] = 2

        if (not in1) or (not in2) or (in1 + in2 != 3):
            continue

        direct = in1 == 2
        selector = 2 if direct else 1
        m = [0] * 3;
        for i in [0, 1, 2]:
            if mask[i] == selector:
                m[2] = i

        m[0] = (m[2] + 1) % 3;
        m[1] = (m[2] + 2) % 3;
        #m[a, b, c] - mapping for triplet
        t.m = m
        t.b = mask

        triplets_list.append(t)

    images = {}
    print("Block1")
    for n,i in images1.items():
        images[n] = i

    print("Block2")
    for n,i in images2.items():
        images[n] = i

    print("Number triplet:", len(triplets_list))

    rng = np.random.default_rng()
    if testRandom:
        rt = rng.choice(triplets_list, size=len(triplets_list)//2, replace=False)
    else:
        rt = triplets_list

    #if we want to shuffle input
    #rng.shuffle(rt)

    rot,tr,u,err = compute_bascule(images, images1, images2, rt)

    np.set_printoptions(suppress=True)
    print('EulerRot', R.from_matrix(rot).as_euler('XYZ', degrees=True))
    print(bcolors.OKCYAN, 'Bascule', rot, bcolors.ENDC)
    print(bcolors.OKBLUE,'Lambda', u, bcolors.ENDC)
    print(bcolors.OKGREEN,'Tr', np.array(tr), bcolors.ENDC)

    if testBascule:
        for n,i in images2.items():
            print('---------------------')
            print('Image: ', n, ':')
            print('DiffRot', R.from_matrix((rot @ i.rot) @
                images2_ori[n].rot.transpose()).as_euler('XYZ', degrees=True))
            print('OriTr', (images2_ori[n].pos) - (u*i.pos+tr))

    if not err and u > 0.:
        #Apply Bascule
        for n,i in images2.items():
            images[n].rot = (rot @ i.rot)
            images[n].pos = u*(rot @ i.pos)+tr

    save_images(ORIOUT_NAME, images.values())

if __name__ == '__main__':
    sys.exit(main())



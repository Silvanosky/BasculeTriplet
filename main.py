#!/bin/python

from re import L
import xml.etree.ElementTree as ET
import sys,os
import parse
from scipy.spatial.transform import Rotation as R
import scipy
import numpy as np


if len(sys.argv) < 4:
    print(sys.argv[0] + " TRIPLETPATH ORI1 OR2 [test|random]")
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

testBascule = False
testRandom = False

if len(sys.argv) == 5:
    if sys.argv[4] == 'test':
        testBascule = True
    if sys.argv[4] == 'random':
        testRandom = True


def gs(X):
    Q, R = np.linalg.qr(X)
    return Q

class Image():
    def __init__(self, name, pos, rot):
        self.name = name
        self.pos = pos
        #self.rot = gs(rot)
        self.rot = rot

    def __str__(self):
        return "(" + self.name + ")" + str(self.pos) + "|" + str(self.rot)

    def inv(self):
        return Image(self.name, -1 * (self.rot@self.pos), self.rot.transpose())

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

"""
Load the MicMac View Orientation XML folder.
"""
def load_images(path, offset_rot = np.identity(3), offset_tr = [0,0,0],
                offset_lamda = 1.):
    images = {}
    with os.scandir(path + '/') as entries:
        for entry in entries:
            if entry.name.startswith("Orientation-"):
                root = ET.parse(path + '/' + entry.name).getroot()
                if "ExportAPERO" == root.tag: #Handle when everything inside
                    root = root.find("OrientationConique")

                c = root.find('Externe')
                pos = xml_load_pos(c)
                rot = xml_load_rot(c.find("ParamRotation").find('CodageMatr'))
                name = parse.parse("Orientation-{}.xml", entry.name)[0]

                images[name] = Image(name, offset_rot @ (offset_lamda *
                    (pos+offset_tr)),
                                     offset_rot @ rot)
    return images

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
                    pos.append(np.array([0.,0.,0.]))
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

"""
Takes a list of images and compute the rotation going from block2 to block1
"""
def compute_rotation(images, triplets):
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

    return mean_rotation(rot).transpose()

def computeall_tr_u(rot, images1, images2, images, triplets):
    if len(triplets) < 2:
        #Need 2 triplet to work ?
        return 0,0

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
    b = np.zeros(n_y, dtype=float)

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
        b[r:r+B.shape[0]] = B

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
        b[r:r+B.shape[0]] = B

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

        n_t += 1

    x, res, rank, s = np.linalg.lstsq(a.astype(float), b.astype(float),
                                      rcond=None)

    np.set_printoptions(suppress=True)
    print("res", res)
    print("rank", rank)
    print("s", s)
    print("x", x)

    e = len(x)
    return [x[e-3], x[e-2], x[e-1]], x[e-4]

def compute_bascule(images, images1, images2, triplets):
    rot = compute_rotation(images, triplets)
    print("Bascule", rot)

    tr,u = computeall_tr_u(rot, images1, images2, images, triplets)

    return rot,tr,u


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
        rt = rng.choice(triplets_list, size=2, replace=False)
    else:
        rt = triplets_list

    #if we want to shuffle input
    #rng.shuffle(rt)

    rot,tr,u = compute_bascule(images, images1, images2, rt)

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

if __name__ == '__main__':
    sys.exit(main())







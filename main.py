#!/bin/python

from re import L
import xml.etree.ElementTree as ET
import sys,os
import parse
from scipy.spatial.transform import Rotation as R
import numpy as np

if len(sys.argv) < 4:
    print(sys.argv[0] + " TRIPLETPATH ORI1 OR2")
    exit(0)

TRIP_NAME = sys.argv[1]
ORI1_NAME = sys.argv[2]
ORI2_NAME = sys.argv[3]

class Image():
    def __init__(self, name, pos, rot):
        self.name = name
        self.pos = pos
        self.rot = rot
    def __str__(self):
        return "(" + self.name + ")" + str(self.pos) + "|" + str(self.rot)

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
def load_images(path, offset_rot = np.identity(3), offset_tr = [0,0,0]):
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

                images[name] = Image(name, pos + offset_tr, offset_rot @ rot)
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
                    rot.append(np.identity(3).astype(np.longdouble))
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

def mean_rotation(rots):

    allrot = np.identity(3).astype(np.longdouble)
    for r in rots:
        allrot = allrot + r
    allrot = allrot * (np.longdouble(1.0) / np.longdouble(len(rots)))

    #u, s, vh = np.linalg.svd(allrot.astype(np.double))
    #ns = np.identity(3)
    #for i in range(3):
    #    if s[i] > 0:
    #        ns[i,i] = 1.
    #    else:
    #        ns[i,i] = -1.
    #return u @ ns @ vh
    return allrot


def compute_rotation(images, triplets):
    rot = []
    for t in triplets:
        #Rotate triplet on B1
        names = [t.names[t.m[0]], t.names[t.m[1]], t.names[t.m[2]]]

        # Orient triplet from V0 and V1
        rot1 = images[names[0]].rot @ t.rot[t.m[0]].transpose()
        rot2 = images[names[1]].rot @ t.rot[t.m[1]].transpose()
        finalrot = mean_rotation([rot1, rot2])
        print("final", finalrot)
        print("--")
        print(R.from_matrix(images[names[0]].rot).as_euler('XYZ', degrees=True))
        print(R.from_matrix(finalrot @ t.rot[t.m[0]]).as_euler('XYZ', degrees=True))
        print("---")

        # Get difference from V3 using rotation V1 and V2
        rotfinal3   = finalrot @ t.rot[t.m[2]] @ images[names[2]].rot.transpose()
        ###############

        print("---")

        # Orient triplet from V3
        rotb = images[names[2]].rot @ t.rot[t.m[2]].transpose()
        # Get difference from V0 using rotation V3
        rotfinal1  = images[names[0]].rot @ (rotb @ t.rot[t.m[0]]).transpose()
        rotfinal2  = images[names[1]].rot @ (rotb @ t.rot[t.m[1]]).transpose()

        print(R.from_matrix(rotfinal3).as_euler('XYZ', degrees=True))
        print(R.from_matrix(rotfinal1).as_euler('XYZ', degrees=True))
        print(R.from_matrix(rotfinal2).as_euler('XYZ', degrees=True))


        print("Bascule", rotfinal3)
        rot.append(rotfinal1.transpose())
        rot.append(rotfinal3.transpose())
        print('-----------------------------')
    return mean_rotation(rot)

def compute_tr_u(images, triplets):
    if len(triplets) != 2:
        #Need 2 triplet to work ?
        return 0,0

    t1 = triplets[0]
    t2 = triplets[1]
    # t_a_B1[3] = a[3] * lambda + t[3]
    a = np.array([
        [float(t1.pos[t1.m[0]][0]),1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [t1.pos[t1.m[0]][1],0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [t1.pos[t1.m[0]][2],0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [t1.pos[t1.m[1]][0],1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [t1.pos[t1.m[1]][1],0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
        [t1.pos[t1.m[1]][2],0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[0]][0],1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[0]][1],0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[0]][2],0.,0.,1.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[1]][0],1.,0.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[1]][1],0.,1.,0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,t2.pos[t2.m[1]][2],0.,0.,1.,0.,0.,0.,0.],
        [-t1.pos[t1.m[2]][0],-1.,0.,0.,0.,0.,0.,0.,images[t1.names[t1.m[2]]].pos[0],1.,0.,0.],
        [-t1.pos[t1.m[2]][1],0.,-1.,0.,0.,0.,0.,0.,images[t1.names[t1.m[2]]].pos[1],0.,1.,0.],
        [-t1.pos[t1.m[2]][2],0.,0.,-1.,0.,0.,0.,0.,images[t1.names[t1.m[2]]].pos[2],0.,0.,1.],
        [0.,0.,0.,0.,-t2.pos[t2.m[2]][0],-1.,0.,0.,images[t2.names[t2.m[2]]].pos[0],1.,0.,0.],
        [0.,0.,0.,0.,-t2.pos[t2.m[2]][1],0.,-1.,0.,images[t2.names[t2.m[2]]].pos[1],0.,1.,0.],
        [0.,0.,0.,0.,-t2.pos[t2.m[2]][2],0.,0.,-1.,images[t2.names[t2.m[2]]].pos[2],0.,0.,1.],

    ])
    b = np.array([
        images[t1.names[t1.m[0]]].pos[0],
        images[t1.names[t1.m[0]]].pos[1],
        images[t1.names[t1.m[0]]].pos[2],
        images[t1.names[t1.m[1]]].pos[0],
        images[t1.names[t1.m[1]]].pos[1],
        images[t1.names[t1.m[1]]].pos[2],
        images[t2.names[t2.m[0]]].pos[0],
        images[t2.names[t2.m[0]]].pos[1],
        images[t2.names[t2.m[0]]].pos[2],
        images[t2.names[t2.m[1]]].pos[0],
        images[t2.names[t2.m[1]]].pos[1],
        images[t2.names[t2.m[1]]].pos[2],
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ])
    x, res, rank, s = np.linalg.lstsq(a, b, rcond=None)

    return [x[9], x[10], x[11]],x[8]

def compute_bascule(images, images1, images2, triplets):
    rot = compute_rotation(images, triplets)
    print("Bascule", rot)
    tr,u = compute_tr_u(images, triplets)

    return rot,tr,u


def main():
    BasculeRot = np.array([[0.9848077, -0.1736482,  0.0000000],
        [-0.0868241, -0.4924039, -0.8660254],
        [0.1503837,  0.8528686, -0.5000000]]).astype(np.longdouble)
    BasculeTr = np.array([130, 42, 12])

    #Load the two block folders
    images1 = load_images(ORI1_NAME)
    print("B2")

    images2 = load_images(ORI2_NAME, BasculeRot, BasculeTr)
    #images2 = load_images(ORI2_NAME)

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
        print("Seclector is ", selector)
        m = [0] * 3;
        for i in [0, 1, 2]:
            if mask[i] == selector:
                m[2] = i

        m[1] = (m[2] + 1) % 3;
        m[0] = (m[2] + 2) % 3;
        t.m = m
        print("M", m)

        triplets_list.append(t)

    images = {}
    print("Block1")
    for n,i in images1.items():
        print(i)
        images[n] = i

    print("Block2")
    for n,i in images2.items():
        print(i)
        images[n] = i

    #print("Selected Triplets")
    #for t in triplets_list:
    #    print(t)

    print("Number triplet:", len(triplets_list))

    rot,tr,u = compute_bascule(images, images1, images2, [triplets_list[0], triplets_list[1]])
    print('DiffRot', R.from_matrix(rot @ BasculeRot.transpose()).as_euler('XYZ', degrees=True))
    print('DiffTr', tr - BasculeTr)
    print('Diffu', u)

main()







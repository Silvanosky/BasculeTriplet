#!/bin/python

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
    def __repr__(self):
        return "Image()"
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
            r += str(self.pos[i])
            r += str(self.rot[i])
            r += "\n"
            i += 1
        return r


def xml_loadPos(e):
    pos = [0] * 3
    i = 0
    for e in e.find("Centre").text.split(" "):
        pos[i] = float(e)
        i += 1
    return pos

def xml_loadRot(e):
    rot = [0] * 9
    j = 0
    for l in ["L1", "L2", "L3"]:
        i = 0
        for c in e.find(l).text.split(" "):
            rot[j * 3 + i] = float(c)
            i += 1
        j += 1
    return rot


def loadImages(path):
    images = []
    with os.scandir(path + '/') as entries:
        for entry in entries:
            if entry.name.startswith("Orientation-"):
                root = ET.parse(path + '/' + entry.name).getroot()
                if "ExportAPERO" == root.tag: #Handle when everything inside
                    root = root.find("OrientationConique")

                c = root.find('Externe')
                pos = xml_loadPos(c)
                rot = xml_loadRot(c.find("ParamRotation").find('CodageMatr'))
                name = parse.parse("Orientation-{}.xml", entry.name)[0]

                images.append(Image(name, pos, rot))
    return images

def loadTripletList(path):
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
                    pos.append([0] * 3)
                    rot = []
                    rot.append([0] * 9)
                    r = ET.parse(path + "/" + names[0] + "/"\
                                 + names[1] + "/Triplet-OriOpt-" + names[2] + ".xml").getroot()
                    pos.append(xml_loadPos(r.find("Ori2On1")))
                    rot.append(xml_loadRot(r.find("Ori2On1").find("Ori")))
                    pos.append(xml_loadPos(r.find("Ori3On1")))
                    rot.append(xml_loadRot(r.find("Ori3On1").find("Ori")))

                    triplets_list.append(Triplet(names, pos, rot))
    return triplets_list

def checkUnique(images1, images2):
    names = set()
    for i in images1:
        names.add(i.name)
    for i in images2:
        if i.name in names:
            return False
    return True


images1 = loadImages(ORI1_NAME)
images2 = loadImages(ORI2_NAME)

if not checkUnique(images1, images2):
    print("Same image in both block")
    exit(1)

simages1 = set()
for i in images1:
    simages1.add(i.name)
simages2 = set()
for i in images2:
    simages2.add(i.name)

triplets_list_all = loadTripletList(TRIP_NAME)
triplets_list = []

for t in triplets_list_all:
    in1 = 0
    in2 = 0
    mask = [0]*3
    for i in [0, 1 , 2]:
        if t.names[i] in simages1:
            in1 += 1
            mask[i] = 1
        if t.names[i] in simages2:
            in2 += 1
            mask[i] = 2

    if (not in1) or (not in2) or (in1 + in2 != 3):
        continue

    direct = in1 == 2
    selector = 1 if direct else 2
    m = [0] * 3;
    for i in [0, 1, 2]:
        if mask[i] != selector:
            m[2] = i
    m[1] = (m[2] + 2) % 3;
    m[0] = (m[2] + 1) % 3;

    triplets_list.append(t)

for t in triplets_list:
    print(t)

print(len(triplets_list))






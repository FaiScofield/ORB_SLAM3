#! /usr/bin/env python
# -*- coding: utf-8 -*-


import sys
from matplotlib import pyplot as plt
import numpy as np

# R21 and t21
def solvICP(odomVI, odomRaw):
    center_e = np.array([[0.], [0.]])
    center_g = np.array([[0.], [0.]])
    for i in range(len(odomVI)):
        pe = np.array([[odomVI[i][0]], [odomVI[i][1]]])
        pg = np.array([[odomRaw[i][0]], [odomRaw[i][1]]])
        center_e = center_e + pe
        center_g = center_g + pg

    # print 'tatal sum:', center_e, center_g
    # line vector
    center_e = center_e / len(odomVI)
    center_g = center_g / len(odomVI)
    # print 'ave:', center_e, center_g

    W = np.mat(np.zeros((2, 2)))
    for i in range(len(odomVI)):
        de = np.array([[odomVI[i][0]], [odomVI[i][1]]]) - center_e
        dg = np.array([[odomRaw[i][0]], [odomRaw[i][1]]]) - center_g
        W = W + np.dot(dg, np.transpose(de))

    # print 'W = ', W
    U, S, VT = np.linalg.svd(W)
    # print 'U, S, V = ', U, S, VT

    R = np.dot(U, VT)
    t = center_g - np.dot(R, center_e)

    return R, t

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    list -- list of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    # if remove_bounds:
    #     lines = lines[100:-100]
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches


if __name__ == '__main__':
    if len(sys.argv) == 5:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
        startIndex = int(sys.argv[3])
        endIndex = int(sys.argv[4])
    elif len(sys.argv) == 3:
        fileVI = sys.argv[1]
        fileOdo = sys.argv[2]
        startIndex = 0
        endIndex = 3000
    else:
        print('Usage: run_exe <trajectoryKF> <odo_raw> [startIndex=0] [endIndex=len(XX)]')
        print('trajectoryKF format: timestamp tx ty tz rx ry rz rw')
        print('odo_raw_file format: timestamp x y yaw')
        sys.exit(0)
    print 'Set startIndex and endIndex to: ', startIndex, endIndex

    # Read VI
    dataVI = read_file_list(fileVI)
    print 'Read all dataVI size: ', len(dataVI)

    # Read odomRaw All
    dataOdom = read_file_list(fileOdo)
    odomData = open(fileOdo, "r")
    print 'Read all odom size: ', len(dataOdom)

    # match
    matches = associate(dataVI, dataOdom, 0, 0.02)
    print 'Get associate matches size: ', len(matches)

    viom = [[float(value) * 1.15 for value in dataVI[a][0:2]] for a,b in matches]
    odom = [[float(value) for value in dataOdom[b][0:2]] for a,b in matches]
    x0 = odom[0][0]
    y0 = odom[0][1]
    for i in range(len(odom)):
        if i == 0:
            continue

        odom[i][0] -= x0 
        odom[i][1] -= y0
    odom[0][0] = odom[0][1] = 0

    # Get odomRaw with VI KF pose
    R, t = solvICP(viom, odom)
    print('R12:', R)
    print('t12:', t)
    print('yaw:', np.arccos(R[0][0]))
    # R12 = np.linalg.inv(R)
    # print('R12:', R12)
    # print('yaw12:', np.arccos(R12[0][0]))
    # theta = 0.9333
    # R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # print('new R:', R)

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    scale = 1
    ################## 210412_hall_2 #######################
    # scale = 0.42
    # deltaR = np.array([[0.97437, 0.22495], [-0.22495, 0.97437]])    # 12degree
    # R = deltaR * R
    ################### 210412_hall_2 #######################
    ################## 201224_hall_2 #######################
    # scale = 0.7
    # deltaR = np.array([[0.96126, -0.27564], [+0.27564, 0.96126]])    # -16degree
    # R = deltaR * R
    ################### 201224_hall_2 #######################
    ################## 201224_indoor #######################
    # scale = 0.65
    # deltaR = np.array([[0.96126, -0.27564], [+0.27564, 0.96126]])    # -16degree
    # R = deltaR * R
    ################### 201224_indoor #######################
    ################## 210412_indoor #######################
    scale = 0.77
    deltaR = np.array([[0.95106, 0.30902], [-0.30902, 0.95106]])    # 18degree
    R = deltaR * R
    ################### 210412_indoor #######################
    ###################  201223_hall  #######################
    # scale = 0.6
    # deltaR = np.array([[0.96126, 0.27564], [-0.27564, 0.96126]])    # 16degree
    # R = deltaR * R
    ###################  201223_hall  #######################
    for i in range(len(viom)):
        # viom[i] = np.dot(R, viom[i]) + t
        p1 = np.array([[viom[i][0]], [viom[i][1]]])
        p2 = (R * p1 + t).tolist()
        # p2 = (R * p1 * 1.15).tolist()
        x1.append(p2[0][0] * scale)
        y1.append(p2[1][0] * scale)
        # x1.append(float(viom[i][0]))
        # y1.append(float(viom[i][1]))

        if i == 0:
            x0 = x1[0]
            y0 = y1[0]
        else:
            x1[i] -= x0 
            y1[i] -= y0
    x1[0] = y1[0] = 0

    for i in range(len(odom)):
        # odom[i] = np.dot(R, np.array(odom[i])) + t
        # p1 = np.array([[odom[i][0]], [odom[i][1]]])
        # p2 = (R * p1 + t).tolist()
        # x2.append(p2[0][0])
        # y2.append(p2[1][0])
        # if i > 140:
        #     lx = float(odom[i - 1][0])
        #     ly = float(odom[i - 1][1])
        #     cx = float(odom[i][0])
        #     cy = float(odom[i][1])
        #     dx = cx - lx
        #     dy = cy - ly
        #     dx = 
        #     # angle = np.arctan(dy / dx * 0.5)
        #     # fx = lx + d * np.cos(angle)
        #     # fy = ly + d * np.sin(angle)
        #     fx = cx + 0.02
        #     fy = cy + 0.011
        #     x2.append(fx)
        #     y2.append(fy)

        x2.append(float(odom[i][0]))
        y2.append(float(odom[i][1]))

    # compute errors
    sum_error = 0
    for i in range(len(viom)):
        e = np.sqrt((x1[i] - x2[i]) * (x1[i] - x2[i]) + (y1[i] - y2[i]) * (y1[i] - y2[i]))
        sum_error = sum_error + e
    print('ave error:', sum_error / len(viom))

    x1[0] = 0
    y1[0] = 0

    plt.plot(x1, y1, color='red', linewidth=1.0, linestyle = '-', label='Trajectory_OURS')
    # plt.plot(x1[0], y1[0], 'ro', label='Start_VI')
    plt.plot(x2, y2, color='black', linewidth=1.0, linestyle = '-', label='Trajectory_ODOM')
    # plt.plot(x2[0], y2[0], 'ko', label='SP_GT')
    plt.plot(x2[0], y2[0], 'ko')

    # x3 = [0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.33]
    # y3 = [0, -0.1,-0.15, -0.18, -0.2, -0.21, -0.22]
    # plt.plot(x3, y3, color='blue', linewidth=1.0, linestyle = '-', label='Trajectory_ORB')
    plt.legend(loc='best')
    # plt.xlim(-1, 2)
    # plt.ylim(-1, 2)
    plt.axis('equal')
    plt.show()


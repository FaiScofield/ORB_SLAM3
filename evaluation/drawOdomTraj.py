
import sys
import numpy
import argparse
import associate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.patches import Ellipse


def plot_traj(ax,sync_stamps,traj,style,color,label):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    sync_stamps -- time sync_stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    sync_stamps.sort()
    interval = numpy.median([s-t for s,t in zip(sync_stamps[1:],sync_stamps[:-1])])
    x = []
    y = []
    last = sync_stamps[0]
    for i in range(len(sync_stamps)):
        if sync_stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label)
            label=""
            x=[]
            y=[]
        last= sync_stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label)
            

if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('odom_file_orig', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('odom_file_sync', help='estimated trajectory (format: timestamp tx ty Rz)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    args = parser.parse_args()

    orig_list = associate.read_file_list(args.odom_file_orig, False)
    sync_list = associate.read_file_list(args.odom_file_sync, False)
    orig_stamps = orig_list.keys()
    sync_stamps = sync_list.keys()
    orig_stamps.sort()
    sync_stamps.sort()

    # set sync_odom start from 0
    sync_odom = [[float(value) for value in sync_list[a][0:3]] for a in sync_stamps]
    x0 = sync_odom[0][0]
    y0 = sync_odom[0][1]
    t0 = sync_odom[0][2]
    max_dis = -1
    max_idx = -1
    for i in range(len(sync_odom)):
        if i == 0:
            continue

        sync_odom[i][0] -= x0
        sync_odom[i][1] -= y0

        if i > 1:
            dx = (sync_odom[i-1][0] - sync_odom[i][0]) * 1000
            dy = (sync_odom[i-1][1] - sync_odom[i][1]) * 1000
            dis = numpy.sqrt(dx * dx + dy * dy)
            # print "#",i,", dis = ", dis

            if dis > max_dis:
                max_dis = dis
                max_idx = i
        
    sync_odom[0][0] = sync_odom[0][1] = sync_odom[0][2] = 0
    print "max dis is #", max_idx, ", max_dis = ", max_dis, 'mm'

    sync_xyz = numpy.matrix(sync_odom).transpose()
    full_xyz = numpy.matrix([[float(value) for value in orig_list[b][0:3]] for b in orig_stamps]).transpose()

    # draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot_traj(ax,sync_stamps,sync_xyz.transpose().A,'-',"black","Trajectory")
    plot_traj(ax,sync_stamps,sync_xyz.transpose().A,'-',"black","odometry_sync")
    plot_traj(ax,orig_stamps,full_xyz.transpose().A,'-',"blue", "odometry_orig")
    ax.legend()
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.axis('equal')
    if args.plot:
        plt.savefig(args.plot,format="png")
    else:
        plt.savefig('odom_traj.png',format="png")
import readsnap as rs
from scms import find_ridge
from Geometry import get_Distance, get_DeltaAngle, get_Direction

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from progressbar import ProgressBar
# from matplotlib.collections import LineCollection
# from time import perf_counter

# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.cluster import DBSCAN
from sklearn import metrics


def get_FilamentDistance(F):
    d = 0
    for i in range(1, len(F)):
        d += get_Distance(F[i-1], F[i])
    return d


def Create_Mesh(gN, D=2, L=0.25):
    if D == 2:
        x, y = np.mgrid[-L:L:gN*1j, -L:L:gN*1j]
        G = np.transpose(np.array([x.ravel(), y.ravel()]))
        return G, x, y
    elif D == 3:
        x, y, z = np.mgrid[-L:L:gN*1j, -L:L:gN*1j, -L:L:gN*1j]
        G = np.transpose(np.array([x.ravel(), y.ravel(), z.ravel()]))
        return G, x, y, z
    else:
        print ("Wrong input dimension D. It is etiher D=2 or D=3.")


def Get_DensityEstimator(X, G, h=0.01, weights=1):
    D, gN = len(X[0]), len(G)

    H = np.eye(D)*h**2
    px = np.zeros(gN)
    pbar = ProgressBar()
    print ("Calulating density estimator:")
    for i in pbar(range(gN)):
        c = multivariate_normal.pdf(X, mean=G[i], cov=H)

        c = c*weights
        px[i] = np.sum(c)/(len(X)*h**D)

    tau = np.sqrt(np.mean(px**2))
    ID = np.where(px > tau)

    return G[ID[0]], px


def Get_Ridge(X, G, h=0.01, weights=1):
    D = len(X[0])
    gN = len(G)
    G = find_ridge(X.reshape((len(X), D, 1)), G.reshape((gN, D, 1)),
                   DD=D, hh=h, eps=1e-07, maxT=1000,
                   wweights=weights)

    return G.reshape(gN, D)


def Write_BruteFilaments(sN, gN, h=0.01, D=3, weights=False, M0=0):
    SnP = "Snap/snap_%03d" % sN                                 # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)             # Subfind Path and file

    SnH = rs.snapshot_header(SnP)                               # Snap Header
    z0, Om_m, Om_l = SnH.redshift, SnH.omega_m, SnH.omega_l     # Redshift, Density paramaters
    E = (Om_m*(1 + z0)**3 + Om_l)**(0.5)                        # Calculate E for minimal cluster mass limit, dependant on reshift
    MLim = M0/E                                                 # Calculate minimal cluster mass limit

    PP = rs.read_sf_block(SfP, 'GPOS')                      # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    C = PP[0]
    PP = (PP - C)/10**5                                     # Shifting system so central galaxy is in center
    if M0 > 0:
        MVir = rs.read_sf_block(SfP, 'MVIR')                        # Mass Virial
        ID = np.where(MVir > MLim)
        print (len(ID[0])/len(MVir))
        PP = PP[ID[0]]
    X = PP[:, :D]                                           # Taking X, Y or all coordinates

    if weights is True:
        W = rs.read_sf_block(SfP, 'MVIR')                   # Virial mass of galaxies
    elif weights is False:
        W = 1

    if D == 2:
        G, x, y = Create_Mesh(gN, D=D)
    elif D == 3:
        G, x, y, z = Create_Mesh(gN, D=D)

    G, px = Get_DensityEstimator(X, G, h=h, weights=W)      # Getting Density estimator and thresholding
    G = Get_Ridge(X, G, h=h, weights=W)                     # Get ridges

    G = G*10**5

    np.save("Subfil/KDE_%03d_%dD" % (sN, D), px)
    np.save("Subfil/BruteFilaments_%03d_%dD" % (sN, D), G)


def Write_wGas_Filament(sN, gN, h=0.01, D=3, weights=False, M0=0):
    SnP = "Snap/snap_%03d" % sN                                 # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)             # Subfind Path and file

    SnH = rs.snapshot_header(SnP)                               # Snap Header
    z0, Om_m, Om_l = SnH.redshift, SnH.omega_m, SnH.omega_l     # Redshift, Density paramaters
    E = (Om_m*(1 + z0)**3 + Om_l)**(0.5)                        # Calculate E for minimal cluster mass limit, dependant on reshift
    MLim = M0/E                                                 # Calculate minimal cluster mass limit

    PP = rs.read_sf_block(SfP, 'GPOS')                      # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    C = PP[0]
    PP = (PP - C)/10**5                                     # Shifting system so central galaxy is in center
    GP = (rs.read_block(SnP, 'POS', parttype=0) - C)/10**5  # Particle Position
    ID = np.where((GP[:, 0] > -0.05) & (GP[:, 0] < 0.05)
                  & (GP[:, 1] > -0.05) & (GP[:, 1] < 0.05)
                  & (GP[:, 2] > -0.05) & (GP[:, 2] < 0.05))
    print (len(ID[0]))
    GP = GP[ID[0]]
    if M0 > 0:
        MVir = rs.read_sf_block(SfP, 'MVIR')                        # Mass Virial
        ID = np.where(MVir > MLim)
        print (len(ID[0])/len(MVir))
        PP = PP[ID[0]]

    if weights is True:
        W = rs.read_sf_block(SfP, 'MVIR')                   # Virial mass of galaxies
    elif weights is False:
        W = 1

    if D == 2:
        G, x, y = Create_Mesh(gN, D=D)
    elif D == 3:
        G, x, y, z = Create_Mesh(gN, D=D)

    G, px = Get_DensityEstimator(GP, G, h=h, weights=W)      # Getting Density estimator and thresholding
    G = Get_Ridge(GP, G, h=h, weights=W)                     # Get ridges

    G = G*10**5

    np.save("Subfil/Test_KDE_%03d_%dD" % (sN, D), px)
    np.save("Subfil/Test_BruteFilaments_%03d_%dD" % (sN, D), G)


def Plot_Ridge2D(sN, gN, h=0.01, D=2, weights=False):
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)         # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')                      # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    C = PP[0]
    PP = (PP - C)/10**5                                     # Shifting system so central galaxy is in center
    X = PP[:, :D]                                           # Taking X, Y or all coordinates

    if weights is True:
        W = rs.read_sf_block(SfP, 'MVIR')                   # Virial mass of galaxies
    elif weights is False:
        W = 1

    # Creating mesh
    if D == 2:
        G, x, y = Create_Mesh(gN, D=D)
    elif D == 3:
        G, x, y, z = Create_Mesh(gN, D=D)

    G, px = Get_DensityEstimator(X, G, h=h, weights=W)      # Getting Density estimator and thresholding
    G = Get_Ridge(X, G, h=h, weights=W)                     # Get ridges

    x = x*10**2
    y = y*10**2
    G = G*10**2

    plt.pcolormesh(x, y, px.reshape(x.shape))
    plt.xlabel('$x[Mpc \\, h^{-1}]$')
    plt.ylabel('$y[Mpc \\, h^{-1}]$')
    plt.scatter(G[:, 0], G[:, 1], s=0.1, c='red', alpha=0.5)
    plt.savefig('Figures/2DRidge_%03d.png' % sN)


def Plot_3DFilament(sN, animation=False):
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)      # Subfind Path and file
    PP = rs.read_sf_block(SfP, 'GPOS')/10**3          # Subfind Position
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3         # Radii Virial
    PP -= PP[0]

    G = np.load("Subfil/BruteFilaments_%03d_%dD.npy" % (sN, 3))/10**3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=1, c='black', alpha=0.05)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    for i in range(100):
        R = RVir[i]
        ax.plot_surface(R*x+PP[i, 0], R*y+PP[i, 1], R*z+PP[i, 2], color='red', alpha=0.4)

    ax.set_xlabel('$x[Mpc/h]$')
    ax.set_ylabel('$y[Mpc/h]$')
    ax.set_zlabel('$z[Mpc/h]$')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])

    plt.savefig('Figures/3DFilament/3DFilament_%03d.png' % sN)
    plt.clf()


def Plot_wGas_3DFilament(sN, animation=False):
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)      # Subfind Path and file
    PP = rs.read_sf_block(SfP, 'GPOS')/10**3          # Subfind Position
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3         # Radii Virial
    PP -= PP[0]

    G = np.load("Subfil/Test_BruteFilaments_%03d_%dD.npy" % (sN, 3))/10**3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=1, c='black', alpha=0.05)

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    for i in range(100):
        R = RVir[i]
        ax.plot_surface(R*x+PP[i, 0], R*y+PP[i, 1], R*z+PP[i, 2], color='red', alpha=0.4)

    ax.set_xlabel('$x[Mpc/h]$')
    ax.set_ylabel('$y[Mpc/h]$')
    ax.set_zlabel('$z[Mpc/h]$')
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])

    plt.savefig('Figures/3DFilament/Test_3DFilament_%03d.png' % sN)
    plt.clf()


"""
def Get_Filaments(sN, M0=300):
    SnP = "Snap/snap_%03d" % sN                                 # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)             # Subfind Path and file

    SnH = rs.snapshot_header(SnP)                               # Snap Header
    z0, Om_m, Om_l = SnH.redshift, SnH.omega_m, SnH.omega_l     # Redshift, Density paramaters
    E = (Om_m*(1 + z0)**3 + Om_l)**(0.5)                        # Calculate E for minimal cluster mass limit, dependant on reshift
    MLim = M0/E                                                 # Calculate minimal cluster mass limit
    print (z0)
    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                    # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    PP -= PP[0]                                                 # Shifting system so central galaxy is in center
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3                  # Radii Virial
    MVir = rs.read_sf_block(SfP, 'MVIR')                        # Mass Virial

    # Find cluster with larger than minimal cluster mass and identify them
    ID = np.where(MVir > MLim)
    print ("Clusters with mass above", MLim, ":", ID[0].size, "out of", MVir.size)
    FoF, N = ID[0], ID[0].size
    PP, RVir, MVir = PP[FoF], RVir[FoF], MVir[FoF]

    G = np.load("Subfil/BruteFilaments_%03d_3D.npy" % sN)/10**3  # Load brute filaments of designated snap
    print ("Number and dimension of points:", G.shape)

    # Eliminate all points inside virial radius of selected clusters
    for i in range(N):
        ID = np.where(get_Distance(G, PP[i]) > RVir[i])
        G = G[ID[0]]

    F, S = [], []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    for i in range(N):
        ax.plot_surface(RVir[i]*x+PP[i, 0], RVir[i]*y+PP[i, 1], RVir[i]*z+PP[i, 2], color='red', alpha=0.4)
        while True:
            pt = G[np.argmin(get_Distance(G, PP[i]))]
            ax.scatter(pt[0], pt[1], pt[2])
            plt.pause(0.0005)
            tempPP, tempR = np.delete(PP, i, axis=0), np.delete(RVir, i, axis=0)
            if (get_Distance(PP[i], pt) - RVir[i]) > np.min(get_Distance(tempPP, pt) - tempR):
                break
            if (get_Distance(PP[i], pt) - RVir[i]) > 1:
                break
            f = np.array([])
            while True:
                #ax.scatter(pt[0], pt[1], pt[2])
                #plt.pause(0.0005)
                idx = np.argmin(get_Distance(G, pt))
                d = get_Distance(G[idx], pt)
                D = get_Distance(tempPP, pt) - tempR

                if len(f) > 1:
                    dfi, dth = get_DeltaAngle(f[-2], pt, G[idx])
                    if dfi > np.pi/2 or dth > np.pi/2:
                        break

                if d > np.min(D):
                    break

                pt = G[idx]
                f = np.append(f.reshape(f.shape[0], 3), pt.reshape(1, 3), axis=0)
                G = np.delete(G, idx, axis=0)
            if len(f) > 1:
                Is = np.argmin(get_Distance(PP, pt))
                if get_Distance(PP[Is], f[-1]) < 1 and Is > i:
                    F.append(f)
                    S.append([i, FoF[Is]])
    F, S = np.array(F), np.array(S)

    np.save("Subfil/FFTF_%03d" % (sN), S)
    np.save("Subfil/Filaments_%03d" % sN, F)
"""


def Get_Filaments(sN, M0=300):
    SnP = "Snap/snap_%03d" % sN                                 # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)             # Subfind Path and file

    SnH = rs.snapshot_header(SnP)                               # Snap Header
    z0, Om_m, Om_l = SnH.redshift, SnH.omega_m, SnH.omega_l     # Redshift, Density paramaters
    E = (Om_m*(1 + z0)**3 + Om_l)**(0.5)                        # Calculate E for minimal cluster mass limit, dependant on reshift
    MLim = M0                                                 # Calculate minimal cluster mass limit
    # print (z0)
    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                    # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    PP -= PP[0]                                                 # Shifting system so central galaxy is in center
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3                  # Radii Virial
    MVir = rs.read_sf_block(SfP, 'MVIR')                        # Mass Virial

    # Find cluster with larger than minimal cluster mass and identify them
    ID = np.where(MVir > MLim)
    print ("Clusters with mass above", MLim, ":", ID[0].size, "out of", MVir.size)
    FoF, N = ID[0], ID[0].size
    PP, RVir, MVir = PP[FoF], RVir[FoF], MVir[FoF]

    G = np.load("Subfil/BruteFilaments_%03d_3D.npy" % sN)/10**3  # Load brute filaments of designated snap
    print ("Number and dimension of points:", G.shape)

    # Eliminate all points inside virial radius of selected clusters
    for i in range(N):
        ID = np.where(get_Distance(G, PP[i]) > RVir[i])
        G = G[ID[0]]

    F, S = [], []

    db = DBSCAN(eps=0.25, min_samples=15).fit(G)
    labels = db.labels_

    for i in range(len(labels)):
        IDF = np.where(labels == i)
        fr = G[IDF[0]]                            # filament random points
        if len(fr) < 2:
            continue
        f0, fr = fr[0], np.delete(fr, 0, axis=0)
        ID = np.argmin(get_Distance(fr, f0))
        f, dir0, fr, f0 = np.array([f0, fr[ID]]), get_Direction(f0, fr[ID]), np.delete(fr, ID, axis=0), fr[ID]

        for i in range(len(fr)):
            ID = np.argmin(get_Distance(fr, f0))
            dir1 = get_Direction(f0, fr[ID])
            if np.dot(dir0, dir1) >= 0:
                f, dir0, fr, f0 = np.append(f.reshape(f.shape[0], 3), fr[ID].reshape(1, 3), axis=0), dir1, np.delete(fr, ID, axis=0), fr[ID]
            else:
                f = np.flip(f, axis=0)
                f, dir0, fr, f0 = np.append(f.reshape(f.shape[0], 3), fr[ID].reshape(1, 3), axis=0), dir1, np.delete(fr, ID, axis=0), fr[ID]

        D1 = np.sqrt((PP[:, 0] - f[0, 0])**2 + (PP[:, 1] - f[0, 1])**2 + (PP[:, 2] - f[0, 2])**2) - RVir
        D2 = np.sqrt((PP[:, 0] - f[-1, 0])**2 + (PP[:, 1] - f[-1, 1])**2 + (PP[:, 2] - f[-1, 2])**2) - RVir
        IC1, IC2 = None, None
        if np.min(D1) < 0.5:
            IC1 = FoF[np.argmin(D1)]
        if np.min(D2) < 0.5:
            IC2 = FoF[np.argmin(D2)]
        F.append(f)
        S.append([IC1, IC2])

    F, S = np.array(F), np.array(S)

    np.save("Subfil/FFTF_%03d" % (sN), S)
    np.save("Subfil/Filaments_%03d" % sN, F)


def Plot_InDepth(sN, FoF, SH=False):
    # SnP = "Snap/snap_%03d" % sN                                     # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)                 # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                        # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3                      # Radii Virial
    PP -= PP[0]                                                     # Shifting system so central galaxy is in center

    F = np.load("Subfil/Filaments_%03d.npy" % sN, allow_pickle=True)
    S = np.load("Subfil/FFTF_%03d.npy" % sN, allow_pickle=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    ID = np.where(S == FoF)
    print ("Snap:", sN)
    ax.plot_surface(RVir[FoF]*x+PP[FoF, 0], RVir[FoF]*y+PP[FoF, 1], RVir[FoF]*z+PP[FoF, 2], color='red', alpha=0.4)
    for i in range(len(ID[0])):
        Is = S[ID[0][i], ID[1][i]-1]
        f = F[ID[0][i]]
        print ("M", Is, ID[0][i], PP[Is, 0], PP[Is, 1], PP[Is, 2])
        IDs = np.where(S == Is)
        if Is is not None:
            ax.plot_surface(RVir[Is]*x+PP[Is, 0], RVir[Is]*y+PP[Is, 1], RVir[Is]*z+PP[Is, 2], color='red', alpha=0.4)
        ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=2, alpha=1)
        if (SH is True) and Is is not None:
            for j in range(len(IDs[0])):
                Is = S[IDs[0][j], IDs[1][j]-1]
                if Is == FoF:
                    continue
                f = F[IDs[0][j]]
                print (Is, IDs[0][j], PP[Is, 0], PP[Is, 1], PP[Is, 2])
                if Is is not None:
                    ax.plot_surface(RVir[Is]*x+PP[Is, 0], RVir[Is]*y+PP[Is, 1], RVir[Is]*z+PP[Is, 2], color='red', alpha=0.4)
                ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=2, alpha=1)

    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-10, 10])
    # plt.savefig('Figures/InDepth/%03d_%02d' % (sN, FoF))
    plt.show()
    plt.clf()


"""
Write_BruteFilaments(41, 100, M0=10)
Write_BruteFilaments(48, 100, M0=10)
Write_BruteFilaments(58, 100, M0=10)
Write_BruteFilaments(72, 100, M0=10)
Write_BruteFilaments(91, 100, M0=10)

Plot_3DFilament(41)
Plot_3DFilament(48)
Plot_3DFilament(58)
Plot_3DFilament(72)
Plot_3DFilament(91)
"""
Get_Filaments(41, M0=10)
Get_Filaments(48, M0=10)
Get_Filaments(58, M0=10)
Get_Filaments(72, M0=10)
Get_Filaments(91, M0=10)


Plot_InDepth(41, 0, SH=True)
Plot_InDepth(48, 0, SH=True)
Plot_InDepth(58, 0, SH=True)
Plot_InDepth(72, 0, SH=True)
Plot_InDepth(91, 0, SH=True)

import readsnap as rs
from scms import find_ridge

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from progressbar import ProgressBar
from matplotlib.collections import LineCollection

def get_Direction(x, y):
    return (y-x)/np.linalg.norm(y-x)


def get_Perpendicular(v):
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('Error: Direction vector is 0.')

    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    a = np.array([0.5, 0.5, -0.5*(v[0] + v[1])/v[2]])

    return a/np.linalg.norm(a)


def get_Intersection(x1, x2, xC, R):
    dir = get_Direction(x1, x2)
    a = np.linalg.norm(dir)**2
    b = 2*np.dot(dir, x2 - xC)
    c = get_Distance(x2, xC)**2 - R**2

    D = b**2 - 4*a*c

    if D <= 0:
        return False
    else:
        return True


def get_Distance(coord, c):
    d, D = coord.shape, c.shape
    # print (d, D)
    if len(d) == 1:
        if D[0] == 2 and d[0] == 2:
            return np.sqrt((coord[0]-c[0])**2 + (coord[1]-c[1])**2)
        elif D[0] == 3 and d[0] == 3:
            return np.sqrt((coord[0]-c[0])**2 + (coord[1]-c[1])**2 + (coord[2]-c[2])**2)
        else:
            print ("Wrong dimensions")

    elif len(d) == 2:
        if D[0] == 2 and d[1] == 2:
            return np.sqrt((coord[:, 0]-c[0])**2 + (coord[:, 1]-c[1])**2)
        elif D[0] == 3 and d[1] == 3:
            return np.sqrt((coord[:, 0]-c[0])**2 + (coord[:, 1]-c[1])**2 + (coord[:, 2]-c[2])**2)
        else:
            print ("Wrong dimensions")
    else:
        print ("Wrong dimensions")


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


def Write_BruteFilaments(sN, gN, h=0.01, D=3, weights=False):
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)         # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')                      # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    C = PP[0]
    PP = (PP - C)/10**5                                     # Shifting system so central galaxy is in center
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
    # plt.hist2d(X[:, 0], X[:, 1], bins=100)
    plt.scatter(G[:, 0], G[:, 1], s=0.1, c='red', alpha=0.5)
    # plt.savefig('Figures/2DRidge_%03d.png' % sN)
    plt.savefig('Figures/Test.png')


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

    plt.savefig('Figures/3DFilament_%03d.png' % sN)
    plt.clf()


def Get_Filaments(sN, M0=300):
    SnP = "Snap/snap_%03d" % sN                          # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)       # Subfind Path and file

    SnH = rs.snapshot_header(SnP)                       # Snap Header
    z0, Om_m, Om_l = SnH.redshift, SnH.omega_m, SnH.omega_l  # Redshift, Density paramaters
    E = (Om_m*(1 + z0)**3 + Om_l)**(0.5)

    MLim = M0/E

    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3              # Radii Virial
    MVir = rs.read_sf_block(SfP, 'MVIR')                    # Mass Virial
    PP -= PP[0]                                             # Shifting system so central galaxy is in center

    G = np.load("Subfil/BruteFilaments_%03d_3D.npy" % sN)/10**3
    print ("Number and dimension of points:", G.shape)

    ID = np.where(MVir > MLim)
    N = len(ID[0])
    FoF = np.linspace(0, len(PP), len(PP) + 1, dtype=int)[ID[0]]
    PP, RVir, MVir = PP[ID[0]], RVir[ID[0]], MVir[ID[0]]
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    """
    for i in range(N):
        R = RVir[i]
        ID = np.where(get_Distance(G, PP[i]) > R)
        G = G[ID[0]]
        # ax.plot_surface(R*x+PP[i, 0], R*y+PP[i, 1], R*z+PP[i, 2], color='red', alpha=0.4)

    F, S = [], []
    for i in range(2):
        while True:
            pt = G[np.argmin(get_Distance(G, PP[i]))]
            tempPP, tempR = np.delete(PP, i, axis=0), np.delete(RVir, i, axis=0)
            if (get_Distance(PP[i], pt) - RVir[i]) > (get_Distance(tempPP, pt) - tempR).any():
                break
            f = np.array([])
            while True:
                idx = np.argmin(get_Distance(G, pt))
                d = get_Distance(G[idx], pt)
                D = get_Distance(tempPP, pt) - tempR
                # print (np.min(D))
                # print (i)
                if d > D.any():
                    break
                if len(f) > 5 and get_Distance(G[idx], pt) > 6*get_FilamentDistance(f)/len(f):
                    # print (d, np.min(D))
                    break
                pt = G[idx]
                f = np.append(f.reshape(f.shape[0], 3), pt.reshape(1, 3), axis=0)
                G = np.delete(G, idx, axis=0)
            if f.shape[0] != 0:
                # if i != np.argmin(get_Distance(PP, pt)):
                F.append(f)
                S.append([i, FoF[np.argmin(get_Distance(PP, pt))]])

    F, S = np.array(F), np.array(S)
    """
    # print (F)
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    ax.set_zlim([-15, 15])
    for f in F:
        ax.scatter(f[:, 0], f[:, 1], f[:, 2], s=2, alpha=1)
    ax.scatter(G[:, 0], G[:, 1], G[:, 2], s=1, c='black', alpha=0.05)
    # plt.axis('off')
    plt.savefig('Figures/Test.png')
    # plt.show()
    plt.clf()
    """
    return F, S


def FilamentDistance(sN, M0=300):
    F, S = Get_Filaments(sN)
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)         # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3              # Radii Virial
    PP -= PP[0]                                             # Shifting system so central galaxy is in center

    D = np.empty(len(F))
    MD = np.empty(len(F))

    s = np.zeros(len(F))
    c = np.zeros(len(F))
    o = np.zeros(len(F))
    r = np.zeros(len(F))

    for i in range(len(F)):
        f = F[i]
        if len(f) == 1:
            continue
        C1, C2 = S[i, 0], S[i, 1]
        D[i] = get_FilamentDistance(F[i])
        MD[i] = get_Distance(PP[C1], PP[C2]) - RVir[C1] - RVir[C2]

        if np.abs(D[i]-MD[i])/MD[i] < 0.1:
            s[i] = 1
        else:
            c[i] = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)

        if get_Intersection(f[1], f[0], PP[C1], RVir[C1]) is False:
            o[i] = 1
        elif get_Intersection(f[-2], f[-1], PP[C2], RVir[C2]) is False:
            o[i] = 1
        # print (C1, C2)

        ax.plot_surface(RVir[C1]*x+PP[C1, 0], RVir[C1]*y+PP[C1, 1], RVir[C1]*z+PP[C1, 2], color='red', alpha=0.4)
        ax.plot_surface(RVir[C2]*x+PP[C2, 0], RVir[C2]*y+PP[C2, 1], RVir[C2]*z+PP[C2, 2], color='red', alpha=0.4)
        ax.scatter(f[:, 0], f[:, 1], f[:, 2])
        plt.show()
        plt.clf()

        if C1 == C2:
            r[i] = 1
    # print (S)
    print (len(F), np.sum(s)/len(F), np.sum(c)/len(F), np.sum(o)/len(F), np.sum(r)/len(F))

    IDs = np.where(s > 0)
    Ds = D[IDs[0]]

    IDc = np.where(c > 0)
    Dc = D[IDc[0]]

    IDo = np.where(o > 0)
    Do = D[IDo[0]]

    IDr = np.where(r > 0)
    Dr = D[IDr[0]]

    DD = [Ds, Dc, Do, Dr]

    w = 0.5
    bin = np.arange(0, 20+w, w)
    pb = bin[1:]-w/2
    c0, b0 = np.histogram(D, bin)

    p = [[] for i in range(4)]
    bot = 0
    for i in range(2):
        c, b = np.histogram(DD[i], bin)
        p[i] = plt.bar(pb, c, width=w, bottom=bot)
        bot += c

    # plt.legend((p[3][0], p[2][0], p[1][0], p[0][0]),
    #           ("Non-pair",
    #            "Off-centre",
    #            "Curved",
    #            "Straight"))

    plt.savefig('Figures/Test.png')
    plt.clf()


# FilamentDistance(41)


def Plot_RadialProfile(sN, fN, M0=300, rmax=1, parttype=0):
    SnP = "Snap/snap_%03d" % sN                             # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)         # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    GP = rs.read_block(SnP, 'POS', parttype=parttype)/10**3  # Particle Position
    MP = rs.read_block(SnP, 'MASS', parttype=parttype)
    if parttype == 0:
        T = rs.read_block(SnP, 'TEMP', parttype=parttype)   # Gas Temperature

    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3              # Radii Virial

    GP, PP = GP - PP[0], PP - PP[0]                         # Shifting system so central galaxy is in center
    F, S = Get_Filaments(41, M0=M0)

    f = F[fN]
    ID = np.where((get_Distance(GP, PP[S[fN, 0]]) > RVir[S[fN, 0]])
                  & (get_Distance(GP, PP[S[fN, 1]]) > RVir[S[fN, 1]])
                  & (GP[:, 0] > np.min(f[:, 0]) - rmax) & (GP[:, 0] < np.max(f[:, 0]) + rmax)
                  & (GP[:, 1] > np.min(f[:, 1]) - rmax) & (GP[:, 1] < np.max(f[:, 1]) + rmax)
                  & (GP[:, 2] > np.min(f[:, 2]) - rmax) & (GP[:, 2] < np.max(f[:, 2]) + rmax))
    IDPP = np.where((PP[:, 0] > np.min(f[:, 0]) - rmax/2) & (PP[:, 0] < np.max(f[:, 0]) + rmax/2)
                    & (PP[:, 1] > np.min(f[:, 1]) - rmax/2) & (PP[:, 1] < np.max(f[:, 1]) + rmax/2)
                    & (PP[:, 2] > np.min(f[:, 2]) - rmax/2) & (PP[:, 2] < np.max(f[:, 2]) + rmax/2))
    GP, MP = GP[ID[0]], MP[ID[0]]
    PP, RVir = PP[IDPP[0]], RVir[IDPP[0]]
    if parttype == 0:
        T = T[ID[0]]

    long = 0

    for i in range(len(f)-1):
        f1, f2 = f[i], f[i+1]
        dir = get_Direction(f1, f2)
        tmax = (f2[0] - f1[0])/dir[0]
        f1_GP = f1 - GP
        t = - np.dot(f1_GP, dir)
        d = np.linalg.norm(f1_GP - np.transpose(t*dir[:, np.newaxis]), axis=1)
        ID = np.where((t < tmax) & (t > 0) & (d < rmax))

        if i == 0:
            GPf, MPf = GP[ID[0]], MP[ID[0]]
            per_dis = d[ID[0]]
            long_dis = long + t[ID[0]]
            if parttype == 0:
                Tf = T[ID[0]]
        else:
            GPf, MPf = np.append(GPf, GP[ID[0]], axis=0), np.append(MPf, MP[ID[0]])
            per_dis = np.append(per_dis, d[ID[0]])
            long_dis = np.append(long_dis, long + t[ID[0]])
            if parttype == 0:
                Tf = np.append(Tf, T[ID[0]])

        GP = np.delete(GP, ID[0], axis=0)
        if parttype == 0:
            T = np.delete(T, ID[0])
        long += get_Distance(f1, f2)

    if parttype == 0:
        WHIM = np.where((Tf > 10**5) & (Tf < 10**7))
        HOT = np.where((Tf > 10**7))
        COLD = np.where((Tf < 10**5))
    """
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(GPf[:, 0], GPf[:, 1], GPf[:, 2], s=1)
    ax.plot(f[:, 0], f[:, 1], f[:, 2], color='red')
    ax.scatter(f[:, 0], f[:, 1], f[:, 2], color='red')
    # ax.plot_surface(RVir[S[fN, 0]]*x+PP[S[fN, 0], 0], RVir[S[fN, 0]]*y+PP[S[fN, 0], 1], RVir[S[fN, 0]]*z+PP[S[fN, 0], 2], color='red', alpha=0.4)
    # ax.plot_surface(RVir[S[fN, 1]]*x+PP[S[fN, 1], 0], RVir[S[fN, 1]]*y+PP[S[fN, 1], 1], RVir[S[fN, 1]]*z+PP[S[fN, 1], 2], color='red', alpha=0.4)
    #ax.set_xlim([-2, 2])
    #ax.set_ylim([1, 6])
    #ax.set_zlim([-1, 5])
    #for i in range(len(RVir)):
    #    R = RVir[i]
    #    ax.plot_surface(R*x+PP[i, 0], R*y+PP[i, 1], R*z+PP[i, 2], color='red', alpha=0.4)
    plt.show()
    plt.clf()
    """
    h = 0.1
    xb = np.insert(np.arange(0.5, rmax + h, h), 0, 0)
    yb = np.linspace(0, long, 20)
    print (yb)
    vol = np.pi*(xb[1:]**2 - xb[:len(xb)-1]**2)

    rho, xb, yb = np.histogram2d(per_dis, long_dis, bins=(xb, yb), weights=MPf)
    print (rho.shape, xb.shape, yb.shape, vol.shape)
    rho = np.transpose(np.transpose(rho/(yb[1:]))/vol)
    m_rho = np.mean(rho)
    plots = [np.column_stack([xb[1:], y]) for y in np.transpose(rho)/m_rho]
    fig, ax = plt.subplots()
    line_segments = LineCollection(plots, cmap=plt.cm.jet)
    line_segments.set_array(yb)
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments)
    axcb.set_label('Distance from 1st FoF')
    # ax.set_xlim(np.min(xb), np.max(xb))
    # ax.set_ylim(np.min(rho), np.max(rho))

    # plt.plot(xb, 10*xb**(-2))
    # plt.plot(xb, 0.1*xb**(-3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.savefig('Figures/Test.png')
    plt.show()
    plt.clf()
    print (RVir[0])
    """
    ID = WHIM[0]
    CW, xm, ym = np.histogram2d(long_dis, per_dis, bins=50,           # With weights
                                normed=False, weights=Tf)
    CC, xm, ym = np.histogram2d(long_dis, per_dis, bins=50,            # Without weights
                                normed=False)
    xm, ym = np.meshgrid(xm, ym, indexing='ij')         # Create meshgrid
    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(xm, ym, CW/CC,
                        cmap=plt.cm.jet)
                        #vmin=10**5, vmax=10**7)                      # Plot
    cbar = fig.colorbar(pcm, ax=ax, ticks=[10**5, 1*10**6, 2*10**6,
                                           3*10**6, 4*10**6,
                                           5*10**6, 6*10**6,
                                           7*10**6, 8*10**6,
                                           9*10**6, 10**7])
    cbar.ax.set_yticklabels(['$0.1$','$1$','$2$','$3$','$4$',
                             '$5$','$6$','$7$','$8$','$9$','$10$'])
    cbar.set_label('$T\,[10^{6}\cdot K]$', rotation=270, labelpad=15)
    plt.colorbar(pcm)
    plt.tight_layout()
    # plt.savefig('Figures/Test.png')
    plt.clf()
    """


Plot_RadialProfile(41, 2, rmax=2, parttype=0)

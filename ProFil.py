import readsnap as rs
from Geometry import get_Distance, get_Direction, get_VectorAngle, get_Intersection
from FindFilament import get_FilamentDistance, Create_Mesh, Get_DensityEstimator, Get_Ridge

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm


def FilamentDistance(sN):
    F = np.load("Subfil/Filaments_%03d.npy" % sN, allow_pickle=True)
    S = np.load("Subfil/FFTF_%03d.npy" % sN, allow_pickle=True)

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

        fia, tha = np.zeros(len(f)-1), np.zeros(len(f)-1)
        for j in range(1, len(f)):
            d = get_Direction(f[j], f[j-1])
            fi, th = get_VectorAngle(d)
            fia[j-1] = fi
            tha[j-1] = th

        if (np.max(fia) - np.min(fia) < 0.25) and (np.max(tha) - np.min(tha) < 0.25):
            s[i] = 1
        else:
            c[i] = 1

        if get_Intersection(f[1], f[0], PP[C1], RVir[C1]) is False:
            o[i] = 1
        elif get_Intersection(f[-2], f[-1], PP[C2], RVir[C2]) is False:
            o[i] = 1
        # print (C1, C2)

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
    #for i in range(2):
    #    c, b = np.histogram(DD[i], bin)
    #    p[i] = plt.bar(pb, c, width=w, bottom=bot)
    #    bot += c

    # plt.legend((p[3][0], p[2][0], p[1][0], p[0][0]),
    #           ("Non-pair",
    #            "Off-centre",
    #            "Curved",
    #            "Straight"))

    plt.hist(D)
    plt.savefig('Figures/Test.png')
    plt.clf()


def Get_MapFilament(sN, fN, fNp=-1, rmax=2, parttype=0):
    SnP = "Snap/snap_%03d" % sN                             # Snap Path and file
    SfP = "Subfind/groups_%03d/sub_%03d" % (sN, sN)         # Subfind Path and file

    PP = rs.read_sf_block(SfP, 'GPOS')/10**3                # Galaxy positions (reducing to lower numbers so exp(PP) is not 0)
    GP = rs.read_block(SnP, 'POS', parttype=parttype)/10**3  # Particle Position
    MP = rs.read_block(SnP, 'MASS', parttype=parttype)
    if parttype == 0:
        T = rs.read_block(SnP, 'TEMP', parttype=parttype)   # Gas Temperature
        Z = rs.read_block(SnP, 'Zs', parttype=0)            # Masses of chemical elements
        CLDX = rs.read_block(SnP, 'CLDX', parttype=0)       # Fraction of gas in cloud
        RHO = rs.read_block(SnP, 'RHO', parttype=0)

    RVir = rs.read_sf_block(SfP, 'RVIR')/10**3              # Radii Virial

    GP, PP = GP - PP[0], PP - PP[0]                         # Shifting system so central galaxy is in center
    F = np.load("Subfil/Filaments_%03d.npy" % sN, allow_pickle=True)

    f = F[fN]
    print (get_FilamentDistance(f))
    if fNp >= 0:
        fs = F[fNp]
        if get_Distance(f[-1], fs[0]) < get_Distance(f[-1], fs[0]):
            f = np.append(f, fs, axis=0)
        else:
            f = np.append(f, np.flip(fs, 0), axis=0)
        print (get_FilamentDistance(f))

    ID = np.where((GP[:, 0] > np.min(f[:, 0]) - rmax) & (GP[:, 0] < np.max(f[:, 0]) + rmax)
                  & (GP[:, 1] > np.min(f[:, 1]) - rmax) & (GP[:, 1] < np.max(f[:, 1]) + rmax)
                  & (GP[:, 2] > np.min(f[:, 2]) - rmax) & (GP[:, 2] < np.max(f[:, 2]) + rmax))
    IDPP = np.where((PP[:, 0] > np.min(f[:, 0]) - rmax/2) & (PP[:, 0] < np.max(f[:, 0]) + rmax/2)
                    & (PP[:, 1] > np.min(f[:, 1]) - rmax/2) & (PP[:, 1] < np.max(f[:, 1]) + rmax/2)
                    & (PP[:, 2] > np.min(f[:, 2]) - rmax/2) & (PP[:, 2] < np.max(f[:, 2]) + rmax/2))
    GP, MP = GP[ID[0]], MP[ID[0]]
    PP, RVir = PP[IDPP[0]], RVir[IDPP[0]]
    if parttype == 0:
        T = T[ID[0]]
        Z = Z[ID[0]]
        CLDX = CLDX[ID[0]]
        ZM = Z[:, 9]/(MP*(1-CLDX)-np.sum(Z, axis=1))/(0.001587)
        RHO = RHO[ID[0]]

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
                Zf = ZM[ID[0]]
                RHOf = RHO[ID[0]]
        else:
            GPf, MPf = np.append(GPf, GP[ID[0]], axis=0), np.append(MPf, MP[ID[0]])
            per_dis = np.append(per_dis, d[ID[0]])
            long_dis = np.append(long_dis, long + t[ID[0]])
            if parttype == 0:
                Tf = np.append(Tf, T[ID[0]])
                Zf = np.append(Zf, ZM[ID[0]])
                RHOf = np.append(RHOf, RHO[ID[0]])
        long += get_Distance(f1, f2)

    if parttype == 0:
        WHIM = np.where((Tf > 10**5) & (Tf < 10**7))
        HOT = np.where((Tf > 10**7))
        COLD = np.where((Tf < 10**5))
        return long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD

    return long_dis, per_dis, GPf, MPf


def Plot_RadialProfile(sN, fN, fNp=-1, rmax=1, parttype=0):
    if parttype == 0:
        long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=parttype)
        if type == 1:
            long_dis, per_dis = long_dis[WHIM[0]], per_dis[WHIM[0]]
            GPf, MPf = GPf[WHIM[0]], MPf[WHIM[0]]
            Tf, Zf, RHOf = Tf[WHIM[0]], Zf[WHIM[0]], RHOf[WHIM[0]]
        elif type == 2:
            long_dis, per_dis = long_dis[HOT[0]], per_dis[HOT[0]]
            GPf, MPf = GPf[HOT[0]], MPf[HOT[0]]
            Tf, Zf, RHOf = Tf[HOT[0]], Zf[HOT[0]], RHOf[HOT[0]]
        elif type == 3:
            long_dis, per_dis = long_dis[COLD[0]], per_dis[COLD[0]]
            GPf, MPf = GPf[COLD[0]], MPf[COLD[0]]
            Tf, Zf, RHOf = Tf[COLD[0]], Zf[COLD[0]], RHOf[COLD[0]]
    else:
        long_dis, per_dis, GPf, MPf = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=parttype)

    long_dis = long_dis[HOT[0]]
    per_dis = per_dis[HOT[0]]
    GPf, MPf = GPf[HOT[0]], MPf[HOT[0]]

    h = 0.1
    R = np.insert(np.arange(0.5, rmax + h, h), 0, 0)
    L = np.linspace(0, long, 40)
    vol = np.pi*(R[1:]**2 - R[:len(R)-1]**2)

    rho, R, L = np.histogram2d(per_dis, long_dis, bins=(R, L), weights=MPf)
    rho = np.transpose(np.transpose(rho/(L[1:] - L[:len(L)-1]))/vol)

    m_rho = 1
    plots = [np.column_stack([R[1:], y]) for y in np.transpose(rho)/m_rho]

    fig, ax = plt.subplots()
    line_segments = LineCollection(plots, cmap=plt.cm.jet)
    line_segments.set_array(L)
    ax.add_collection(line_segments)

    cbaxes = fig.add_axes([0.1375, 0.9375, 0.75, 0.01])
    axcb = fig.colorbar(line_segments, orientation="horizontal", cax=cbaxes)
    axcb.set_label("$L \\quad [Mpc \\, h^{-1}]$", labelpad=-35, y=0.45)

    # ax.set_xlim(np.min(xb), np.max(xb))
    # ax.set_ylim(10**(-5), np.max(rho))

    # ax.plot(R, 10*R**(-2))
    # ax.plot(R, 0.1*R**(-3))

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel("$r \\quad [Mpc \\, h^{-1}]$")
    ax.set_xticks([0.5, 0.6, 1, 2])
    ax.set_xticklabels(["0.5", "", "1", "2"])
    ax.set_ylabel("$\\rho \\quad [10^{10} M_{\\odot} / Mpc^{3}]$")
    ax.set_yticks([1, 10])
    ax.set_yticklabels(["1", "10"])

    plt.savefig('Figures/FilamentProfile_%03d_%03d.png' % (sN, fN))
    plt.clf()


def Plot_TemperatureFilament(sN, fN, fNp=-1, rmax=2, type=0):
    long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=0)

    if type == 1:
        long_dis, per_dis = long_dis[WHIM[0]], per_dis[WHIM[0]]
        GPf, MPf = GPf[WHIM[0]], MPf[WHIM[0]]
        Tf, Zf, RHOf = Tf[WHIM[0]], Zf[WHIM[0]], RHOf[WHIM[0]]
    elif type == 2:
        long_dis, per_dis = long_dis[HOT[0]], per_dis[HOT[0]]
        GPf, MPf = GPf[HOT[0]], MPf[HOT[0]]
        Tf, Zf, RHOf = Tf[HOT[0]], Zf[HOT[0]], RHOf[HOT[0]]
    elif type == 3:
        long_dis, per_dis = long_dis[COLD[0]], per_dis[COLD[0]]
        GPf, MPf = GPf[COLD[0]], MPf[COLD[0]]
        Tf, Zf, RHOf = Tf[COLD[0]], Zf[COLD[0]], RHOf[COLD[0]]

    CW, xm, ym = np.histogram2d(long_dis, per_dis, bins=25,           # With weights
                                normed=False, weights=Tf)

    CC, xm, ym = np.histogram2d(long_dis, per_dis, bins=25,            # Without weights
                                normed=False)

    xm, ym = np.meshgrid(xm, ym, indexing='ij')         # Create meshgrid

    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(xm, ym, CW/CC,
                        cmap=plt.cm.jet)                      # Plot

    cbar = fig.colorbar(pcm, ax=ax)

    cbar.set_label('$T\\,[K]$', rotation=270, labelpad=15)
    plt.ylabel('$r \\, [Mpc/h]$')
    plt.xlabel('$d \\, [Mpc/h]$')
    plt.savefig('Figures/TemperatureFilament_%03d_%02d.png' % (sN, fN))
    plt.clf()

    CW, xm, ym = np.histogram2d(long_dis, per_dis, bins=25,           # With weights
                                normed=False, weights=Zf)

    CC, xm, ym = np.histogram2d(long_dis, per_dis, bins=25,            # Without weights
                                normed=False)

    xm, ym = np.meshgrid(xm, ym, indexing='ij')         # Create meshgrid

    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(xm, ym, CW/CC,
                        cmap=plt.cm.jet, vmax=1)                      # Plot

    cbar = fig.colorbar(pcm, ax=ax)

    cbar.set_label('$Z\\,[Z_{\\odot}]$', rotation=270, labelpad=15)
    plt.tight_layout()
    plt.ylabel('$r \\, [Mpc/h]$')
    plt.xlabel('$d \\, [Mpc/h]$')
    plt.savefig('Figures/MetalicityFilament_%03d_%02d.png' % (sN, fN))
    plt.clf()


def Plot_DensityFilament(sN, fN, fNp=-1, rmax=2, parttype=0, type=0):
    if parttype == 0:
        long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=parttype)
        if type == 1:
            long_dis, per_dis = long_dis[WHIM[0]], per_dis[WHIM[0]]
            GPf, MPf = GPf[WHIM[0]], MPf[WHIM[0]]
            Tf, Zf, RHOf = Tf[WHIM[0]], Zf[WHIM[0]], RHOf[WHIM[0]]
        elif type == 2:
            long_dis, per_dis = long_dis[HOT[0]], per_dis[HOT[0]]
            GPf, MPf = GPf[HOT[0]], MPf[HOT[0]]
            Tf, Zf, RHOf = Tf[HOT[0]], Zf[HOT[0]], RHOf[HOT[0]]
        elif type == 3:
            long_dis, per_dis = long_dis[COLD[0]], per_dis[COLD[0]]
            GPf, MPf = GPf[COLD[0]], MPf[COLD[0]]
            Tf, Zf, RHOf = Tf[COLD[0]], Zf[COLD[0]], RHOf[COLD[0]]
    else:
        long_dis, per_dis, GPf, MPf = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=parttype)

    #plt.hist(per_dis)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(GPf[:, 0], GPf[:, 1], GPf[:, 2], s=1, c='black', alpha=0.05)
    plt.show()
    plt.clf()
    """
    CW, xm, ym = np.histogram2d(long_dis, per_dis, bins=25,           # With weights
                                normed=False)

    xm, ym = np.meshgrid(xm, ym, indexing='ij')         # Create meshgrid

    fig, ax = plt.subplots(1, 1)
    pcm = ax.pcolormesh(xm, ym, CW,
                        cmap=plt.cm.jet, vmax=500)                      # Plot

    fig.colorbar(pcm, ax=ax)
    # cbar.set_label('$T\\,[K]$', rotation=270, labelpad=15)
    # plt.colorbar(pcm)
    # plt.tight_layout()
    plt.ylabel('$r \\, [Mpc/h]$')
    plt.xlabel('$d \\, [Mpc/h]$')
    if parttype == 0:
        plt.savefig('Figures/GasDensityFilament_%03d_%02d.png' % (sN, fN))
    elif parttype == 1:
        plt.savefig('Figures/DMDensityFilament_%03d_%02d.png' % (sN, fN))
    elif parttype == 4:
        plt.savefig('Figures/StarDensityFilament_%03d_%02d.png' % (sN, fN))
    plt.clf()


def Plot_PhaseDiagram_Filament(sN, fN, fNp=-1, rmax=2, bins=100):
    SnP = "Snap/snap_%03d" % sN                             # Snap Path and file
    SnH = rs.snapshot_header(SnP)                               # Snap Header
    z0 = SnH.redshift                                       # Redshift, Density paramaters

    long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=0)

    CW, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # With weights
                              normed=False,
                              weights=Zf)
    CC, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # Without weights
                              normed=False)

    x, y = np.meshgrid(x, y, indexing='ij')             # Creat meshgrid (x----> log10(Rho), y----> log10(T))

    plt.pcolormesh(x, y, CW/CC, cmap=plt.cm.jet, norm=LogNorm(vmax=5, vmin=10**(-5)))   # Plot
    # plt.colorbar()
    plt.title("$z = %.2f $" % (np.abs(z0)))
    plt.xlim([-10, -3])
    plt.ylim([2, 9])
    plt.xlabel("$\\log\\rho\\, [10^{10} \\frac{M_{\\odot}}{kpc^{3}}]$")
    plt.ylabel("$\\log T\\,[K]$")
    plt.tight_layout()
    cbar = plt.colorbar(ticks=[5, 1, 10**(-1), 10**(-2), 10**(-3), 10**(-4), 10**(-5)], extend='both')
    cbar.ax.set_yticklabels(['5', '1', '-1', '-2', '-3', '-4', '-5'])
    cbar.set_label('$\\logZ \\, [Z_{\\odot}]$', rotation=270, labelpad=15)
    plt.savefig('Figures/Filament_Phase_%03d_%03d.png' % (sN, fN))
    plt.clf()

    CW, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # With weights
                              normed=False,
                              weights=long_dis)
    CC, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # Without weights
                              normed=False)

    x, y = np.meshgrid(x, y, indexing='ij')             # Creat meshgrid (x----> log10(Rho), y----> log10(T))

    plt.pcolormesh(x, y, CW/CC, cmap=plt.cm.jet)   # Plot
    # plt.colorbar()
    plt.title("$z = %.2f $" % (np.abs(z0)))
    plt.xlim([-10, -3])
    plt.ylim([2, 9])
    plt.xlabel("$\\log\\rho\\, [10^{10} \\frac{M_{\\odot}}{kpc^{3}}]$")
    plt.ylabel("$\\log T\\,[K]$")
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_label('$d \\, [Mpc/h]$', rotation=270, labelpad=15)
    plt.savefig('Figures/Filament_Phase_FC_%03d_%03d.png' % (sN, fN))
    plt.clf()

    CW, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # With weights
                              normed=False,
                              weights=per_dis)
    CC, x, y = np.histogram2d(np.log10(RHOf), np.log10(Tf), bins=bins,           # Without weights
                              normed=False)

    x, y = np.meshgrid(x, y, indexing='ij')             # Creat meshgrid (x----> log10(Rho), y----> log10(T))

    plt.pcolormesh(x, y, CW/CC, cmap=plt.cm.jet)   # Plot
    # plt.colorbar()
    plt.title("$z = %.2f $" % (np.abs(z0)))
    plt.xlim([-10, -3])
    plt.ylim([2, 9])
    plt.xlabel("$\\log\\rho\\, [10^{10} \\frac{M_{\\odot}}{kpc^{3}}]$")
    plt.ylabel("$\\log T\\,[K]$")
    plt.tight_layout()
    cbar = plt.colorbar()
    cbar.set_label('$r \\, [Mpc/h]$', rotation=270, labelpad=15)
    plt.savefig('Figures/Filament_Phase_FF_%03d_%03d.png' % (sN, fN))
    plt.clf()


def Fine_Tune(sN, fN, gN, fNp=-1, rmax=2):
    long_dis, per_dis, GPf, MPf, Tf, Zf, RHOf, WHIM, HOT, COLD = Get_MapFilament(sN, fN, fNp=fNp, rmax=rmax, parttype=0)

    xmin, xmax = np.min(GPf[:, 0]) - 0.5, np.max(GPf[:, 0]) + 0.5
    ymin, ymax = np.min(GPf[:, 1]) - 0.5, np.max(GPf[:, 1]) + 0.5
    zmin, zmax = np.min(GPf[:, 2]) - 0.5, np.max(GPf[:, 2]) + 0.5

    x, y, z = np.mgrid[xmin:xmax:gN*1j, ymin:ymax:gN*1j, zmin:zmax:gN*1j]
    G = np.transpose(np.array([x.ravel(), y.ravel(), z.ravel()]))

    G, px = Get_DensityEstimator(GPf, G)      # Getting Density estimator and thresholding
    G = Get_Ridge(GPf, G, h=0.5)

    F = np.load("Subfil/Filaments_%03d.npy" % sN, allow_pickle=True)
    f = F[fN]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    ax.scatter(G[:, 0], G[:, 1], G[:, 2])
    ax.scatter(f[:, 0], f[:, 1], f[:, 2])
    ax.set_xlim([np.min(GPf[:, 0]) - 0.5, np.max(GPf[:, 0]) + 0.5])
    ax.set_ylim([np.min(GPf[:, 1]) - 0.5, np.max(GPf[:, 1]) + 0.5])
    ax.set_zlim([np.min(GPf[:, 2]) - 0.5, np.max(GPf[:, 2]) + 0.5])
    plt.show()


# Fine_Tune(41, 2, 25, fNp=25, rmax=5)

Plot_PhaseDiagram_Filament(41, 2, fNp=27, rmax=2)
Plot_PhaseDiagram_Filament(48, 3, fNp=24, rmax=2)
Plot_PhaseDiagram_Filament(57, 0, fNp=26, rmax=2)
Plot_PhaseDiagram_Filament(72, 3, fNp=23, rmax=2)
Plot_PhaseDiagram_Filament(91, 1)


pt = 4
t = 0
"""
# Plot_RadialProfile(91, 2, rmax=2, parttype=0)
Plot_DensityFilament(41, 2, fNp=27, rmax=2, parttype=pt)
Plot_DensityFilament(48, 3, fNp=24, rmax=2, parttype=pt)
Plot_DensityFilament(57, 0, fNp=26, rmax=2, parttype=pt)
Plot_DensityFilament(72, 3, fNp=21, rmax=2, parttype=pt)
Plot_DensityFilament(91, 4, rmax=2, parttype=pt)

Plot_TemperatureFilament(41, 2, fNp=27, rmax=2, type=t)
Plot_TemperatureFilament(48, 3, fNp=24, rmax=2, type=t)
Plot_TemperatureFilament(57, 0, fNp=26, rmax=2, type=t)
Plot_TemperatureFilament(72, 3, fNp=21, rmax=2, type=t)
Plot_TemperatureFilament(91, 4, rmax=2, type=t)
"""

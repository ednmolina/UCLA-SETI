from setiS18.sql import get_data
import setiS18.figures.tfdiagram as tf
from setiS18 import CandidateSignal
import matplotlib.pyplot as plt
import corner
import numpy as np
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

def ClusterCandidates(Parameter1, Parameter2, n_clusters):
    """ Returns the coordinates and labels of the center of the clusters using KMeans algorithms.

        Parameters
        ----------
        Parameter1, Parameter2 : Pandas dataframe
        n_clusters : int (number of clusters wanted)

        """
    StackedParameters = np.vstack((Parameter1,Parameter2))
    #Cluster_Centers = k_means(StackedParameters.T,n_clusters=n_clusters)
    Cluster_Centers = KMeans(n_clusters=n_clusters).fit(StackedParameters.T)
    ClusterLabels = Cluster_Centers.labels_
    ClusterCenters = Cluster_Centers.cluster_centers_
#    print ("Length of the labels is:", len(ClusterLabels))
    x, y = ClusterCenters.T[0], ClusterCenters.T[1]
    return x, y, ClusterLabels

def NormalizeFeature(Data):
    NormalizedData = (Data-np.average(Data))/np.max(Data-np.average(Data))
    return NormalizedData

def StackData(CandArray, Normalize):
    if Normalize==False:
        Frequencies4Y = CandArray['FREQ']
        DriftRate4Y = CandArray['DFDT']
        SNR4Y = CandArray['Z']
        Data4Y = np.vstack((Frequencies4Y, DriftRate4Y, SNR4Y)).T
        return Data4Y
    else:
        Frequencies4Y = NormalizeFeature(CandArray['FREQ'])
        DriftRate4Y = NormalizeFeature(CandArray['DFDT'])
        SNR4Y = NormalizeFeature(CandArray['Z'])
        Data4Y = np.vstack((Frequencies4Y, DriftRate4Y, SNR4Y)).T
        return Data4Y


def GetDataAndCornerPlot(Freq_begin, Freq_end, NormalizationBool, CornerPlotBool):
    """This retrieves the four different sets of data gives a corner plot if wanted

        Parameters
        ----------
        Freq_begin, Freq_end : user input of the frequency range desired
        NormalizationBool, CornerPlotBool : boolean, True for normalized data in scatter and True for a corner plot

    """

    "Get the data"
    Cands4Y = get_data(flag4='Y', frange=(Freq_begin, Freq_end))  # pandas dataframe
    Cands1N2N = get_data(flag1='N', flag2='N', flag3='Y', frange=(Freq_begin, Freq_end))  # pandas dataframe
    Cands1Y2N = get_data(flag1='Y', flag2='N', flag3='Y', frange=(Freq_begin, Freq_end))  # pandas dataframe
    Cands1N2Y = get_data(flag1='N', flag2='Y', flag3='Y', frange=(Freq_begin, Freq_end))  # pandas dataframe

    "Aggregate the data in to large arrays for the corner plot"
    StackData4Y = StackData(Cands4Y, NormalizationBool)
    StackData1N2N = StackData(Cands1N2N, NormalizationBool)
    StackData1Y2N = StackData(Cands1Y2N, NormalizationBool)
    StackData1N2Y = StackData(Cands1N2Y, NormalizationBool)

    if CornerPlotBool == True:
        "Plotting the histograms"
        bins = 120

        plt.figure(figsize=(7, 5))
        figure4Y = corner.corner(StackData4Y, labels=["Freq", "Drift Rate", "SNR"],
                                 color='#FD9827', bins=150,
                                 show_titles=True, verbose=True, plot_contours=True)
        figure1N2N = corner.corner(StackData1N2N, labels=["Freq", "Drift Rate", "SNR"],
                                   color='#651297', bins=150,
                                   show_titles=True, verbose=True, plot_contours=True, fig=figure4Y)
        figure1Y2N = corner.corner(StackData1Y2N, labels=["Freq", "Drift Rate", "SNR"],
                                   color='#149839', bins=150,
                                   show_titles=True, verbose=True, plot_contours=True, fig=figure1N2N)
        corner.corner(StackData1N2Y, labels=["Freq", "Drift Rate", "SNR"], color='#CA0813',
                      bins=150, show_titles=True,
                      verbose=True, plot_contours=True, fig=figure1N2N, alpha=.05)
        plt.show()

    return Cands4Y, Cands1N2N, Cands1Y2N, Cands1N2Y


def ScatterWithMeans(CandFreq, CandDFDT, CandSNR, ClusterBool, nclusterfreq, nclusterdfdt, nclustersnr):
    plt.close()
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(
    20, 20))  # sharex='col', sharey='row')
    ax1.scatter(CandFreq, CandSNR, s=1)
    ax1.set_xlabel('Freq')
    ax1.set_ylabel('SNR')
    ax2.scatter(CandDFDT, CandSNR, s=1)
    ax2.set_xlabel('DFDT')
    ax2.set_ylabel('SNR')
    ax3.hist(CandSNR, bins=100)
    ax3.set_title('SNR Histogram')
    ax4.scatter(CandFreq, CandDFDT, s=1)
    ax4.set_ylabel('DFDT')
    ax5.hist(CandDFDT, bins=100)
    ax5.set_title('DFDT Histogram')
    ax6.scatter(NormalizeFeature(CandDFDT), NormalizeFeature(CandSNR), s=1)
    ax6.set_xlabel('DFDT')
    ax6.set_ylabel('SNR')
    ax6.set_title('Normalized DFDT vs SNR')
    ax7.hist(CandFreq, bins=100)
    ax7.set_title('Freq Histogram')
    ax8.scatter(NormalizeFeature(CandFreq), NormalizeFeature(CandDFDT), s=1)
    ax8.set_xlabel('Freq')
    ax8.set_ylabel('DFDT')
    ax8.set_title('Normalized Freq vs DFDT')
    ax9.scatter(NormalizeFeature(CandFreq), NormalizeFeature(CandSNR), s=1)
    ax9.set_xlabel('Freq')
    ax9.set_ylabel('SNR')
    ax9.set_title('Normalized Freq vs SNR')

    if ClusterBool == True:
    "Cluster algorithm on the normalized data"
        xfeq, yfreq, labelfreq = ClusterCandidates(NormalizeFeature(CandFreq), NormalizeFeature(CandDFDT), nclusterfreq)
        ax8.scatter(xfeq, yfreq)

        xdfdt, ydfdt, labeldfdt = ClusterCandidates(NormalizeFeature(CandFreq), NormalizeFeature(CandSNR), nclusterdfdt)
        ax9.scatter(xdfdt, ydfdt)

        xsnr, ysnr, labelsnr = ClusterCandidates(NormalizeFeature(CandDFDT), NormalizeFeature(CandSNR), nclustersnr)
        ax6.scatter(xsnr, ysnr)

    plt.show()

    return labelfreq, labeldfdt, labelsnr

def VStackPairs(Parameter1, Parameter2):
    StackedParameters = np.vstack((Parameter1,Parameter2))
    return StackedParameters


"Generalize the code"


def Scatter(FreqBegin, FreqEnd, NormalizeBool, CornerPlotBool, NumCluster_Freq, NumCluster_DFDT, NumCluster_SNR):
    "Get the Data"
    Cands4Y, Cands1N2N, Cands1Y2N, Cands1N2Y = GetDataAndCornerPlot(int(FreqBegin), int(FreqEnd),
                                                                    NormalizeBool, CornerPlotBool)

    "Generate the Labels for the candidate signals"
    labelfreq4Y, labeldfdt4Y, labelsnr4Y = ScatterWithMeans(Cands4Y['FREQ'],
                                                            Cands4Y['DFDT'],
                                                            Cands4Y['Z'],True, 3, 2, 3)
    labelfreq1N2N, labeldfdt1N2N, labelsnr1N2N = ScatterWithMeans(Cands1N2N['FREQ'],
                                                                  Cands1N2N['DFDT'],
                                                                  Cands1N2N['Z'],
                                                                  True,
                                                                  NumCluster_Freq,
                                                                  NumCluster_DFDT,
                                                                  NumCluster_SNR)
    labelfreq1Y2N, labeldfdt1Y2N, labelsnr1Y2N = ScatterWithMeans(Cands1Y2N['FREQ'],
                                                                  Cands1Y2N['DFDT'],
                                                                  Cands1Y2N['Z'],
                                                                  True,
                                                                  NumCluster_Freq,
                                                                  NumCluster_DFDT,
                                                                  NumCluster_SNR)
    labelfreq1N2Y, labeldfdt1N2Y, labelsnr1N2Y = ScatterWithMeans(Cands1N2Y['FREQ'],
                                                                  Cands1N2Y['DFDT'],
                                                                  Cands1N2Y['Z'],
                                                                  True,
                                                                  NumCluster_Freq,
                                                                  NumCluster_DFDT,
                                                                  NumCluster_SNR)

    Cands4Y['Frq4YLabel'] = labelfreq4Y.T
    Cands4Y['DFDT4YLabel'] = labeldfdt4Y.T
    Cands4Y['SNR4YLabel'] = labelsnr4Y.T

    Cands1N2N['Frq1N2NLabel'] = labelfreq1N2N.T
    Cands1N2N['DFDT1N2NLabel'] = labeldfdt1N2N.T
    Cands1N2N['SNR1N2NLabel'] = labelsnr1N2N.T

    Cands1Y2N['Frq1Y2NLabel'] = labelfreq1Y2N.T
    Cands1Y2N['DFDT1Y2NLabel'] = labeldfdt1Y2N.T
    Cands1Y2N['SNR1Y2NLabel'] = labelsnr1Y2N.T

    Cands1N2Y['Frq1N2YLabel'] = labelfreq1N2Y.T
    Cands1N2Y['DFDT1N2YLabel'] = labeldfdt1N2Y.T
    Cands1N2Y['SNR1N2YLabel'] = labelsnr1N2Y.T

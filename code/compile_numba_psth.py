from numba.pycc import CC
import numpy as np
import numpy.typing as npt

cc = CC('numba_psth')
cc.output_dir = '/code'
cc.verbose = True


@cc.export('makePSTH', '(f8[:], f8[:], f8, f8, f8, f8)')
def makePSTH(
    spikeTimes: npt.NDArray[np.floating],
    startTimes: npt.NDArray[np.floating],
    baselineDur: float = 0.1,
    responseDur: float = 1.0,
    binSize: float = 0.001,
    convolution_kernel: float = 0.01,
):
    spikeTimes = spikeTimes.flatten()
    startTimes = startTimes - (baselineDur + convolution_kernel / 2)
    windowDur = responseDur + baselineDur + convolution_kernel
    bins = np.arange(0, baselineDur + responseDur + binSize, binSize)
    convkernel = np.ones(int(convolution_kernel / binSize))
    counts = np.zeros(bins.size - 1)
    for i, startTime in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes, startTime)
        endInd = np.searchsorted(spikeTimes, startTime + windowDur)
        counts = counts + np.histogram(spikeTimes[startInd:endInd] - startTime, bins)[0]
    counts = counts / startTimes.size
    counts = np.convolve(counts, convkernel) / (binSize * convkernel.size)
    return (
        counts[convkernel.size - 1 : -convkernel.size],
        bins[: -convkernel.size - 1] - baselineDur,
    )

if __name__ == "__main__":
    cc.compile()
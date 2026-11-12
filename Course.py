import random
import math


class FunRand:
    @staticmethod
    def Exp(timeMean):
        a = 0.0
        while a == 0.0:
            a = random.random()
        return -timeMean * math.log(a)

    @staticmethod
    def Erlang(k, lambd):
        p = 1.0
        for _ in range(k):
            a = 0.0
            while a == 0.0:
                a = random.random()
            p *= a
        return -math.log(p) / lambd

    @staticmethod
    def Norm(timeMean, timeDeviation):
        val = -1.0
        while val <= 0:
            val = random.gauss(timeMean, timeDeviation)
        return val

    @staticmethod
    def Unif(timeMin, timeMax):
        a = 0.0
        while a == 0.0:
            a = random.random()
        return timeMin + a * (timeMax - timeMin)


class Process:
    def __init__(self, name, channelsCnt, maxQ=float('inf')):
        self.name = name
        self.channelCount = channelsCnt
        self.maxQueue = maxQ
        self.queue = 0
        self.channels = [0] * channelsCnt
        self.tnextChannels = [float('inf')] * channelsCnt
        self.tnext = float('inf')
        self.tcurr = 0.0

        self.delayMean = 0.0
        self.delayDev = 0.0
        self.distribution = ""
        self.nextElement = None

        self.meanQueueArea = 0.0
        self.meanBusyArea = 0.0
        self.servedCount = 0

    def setDelay(self, mean, dev=0.0, dist=""):
        self.delayMean = mean
        self.delayDev = dev
        self.distribution = dist.lower()

    def getDelay(self):
        if self.distribution == "exp": return FunRand.Exp(self.delayMean)
        if self.distribution == "norm": return FunRand.Norm(self.delayMean, self.delayDev)
        if self.distribution == "unif": return FunRand.Unif(self.delayMean, self.delayDev)
        if self.distribution == "erlang": return FunRand.Erlang(int(self.delayMean), self.delayDev)
        if self.distribution == "const": return self.delayMean
        return self.delayMean

    def getTnext(self):
        if self.channelCount > 0:
            return min(self.tnextChannels)
        return self.tnext

    def setTcurr(self, t):
        self.tcurr = t

    def getFreeChannel(self):
        for i in range(self.channelCount):
            if self.channels[i] == 0:
                return i
        return -1

    def inAct(self):
        freeChannel = self.getFreeChannel()
        if freeChannel != -1:
            self.channels[freeChannel] = 1
            self.tnextChannels[freeChannel] = self.tcurr + self.getDelay()
        else:
            if self.queue < self.maxQueue:
                self.queue += 1

    def outAct(self):
        channels_to_free = []
        for i in range(self.channelCount):
            if math.isclose(self.tnextChannels[i], self.tcurr, abs_tol=1e-9):
                channels_to_free.append(i)

        for i in channels_to_free:
            self.tnextChannels[i] = float('inf')
            self.channels[i] = 0
            self.servedCount += 1

            if self.nextElement is not None:
                self.nextElement.inAct()

            if self.queue > 0:
                self.queue -= 1
                self.channels[i] = 1
                self.tnextChannels[i] = self.tcurr + self.getDelay()

    def doStatistics(self, delta):
        self.meanQueueArea += self.queue * delta
        self.meanBusyArea += sum(self.channels) * delta


class LoaderProcess(Process):
    def __init__(self, name, earthQ, truckQ, numLoaders=2):
        super().__init__(name, numLoaders, float('inf'))
        self.earthQueueElement = earthQ
        self.truckQueueElement = truckQ
        self.loaderDelays = [14.0 if i == 0 else 12.0 for i in range(numLoaders)]
        self.restTimes = [float('inf')] * numLoaders
        self.isResting = [0] * numLoaders
        self.areaIdle = [0.0] * numLoaders
        self.areaBusy = [0.0] * numLoaders

    def getTnext(self):
        min_t = super().getTnext()
        min_rest = min(self.restTimes) if self.channelCount > 0 else float('inf')
        return min(min_t, min_rest)

    def checkAndStart(self):
        while self.truckQueueElement.queue > 0 and self.earthQueueElement.queue >= 2:
            freeIdx = -1
            for i in range(self.channelCount):
                if (self.earthQueueElement.queue >= 2) and (self.channels[i] == 0) and (self.isResting[i] == 0):
                    freeIdx = i
                    break

            if freeIdx != -1:
                self.earthQueueElement.queue -= 2
                self.truckQueueElement.queue -= 1
                self.channels[freeIdx] = 1
                self.tnextChannels[freeIdx] = self.tcurr + FunRand.Exp(self.loaderDelays[freeIdx])
            else:
                break

    def outAct(self):
        for i in range(self.channelCount):
            if math.isclose(self.restTimes[i], self.tcurr, abs_tol=1e-9):
                self.restTimes[i] = float('inf')
                self.isResting[i] = 0

        for i in range(self.channelCount):
            if math.isclose(self.tnextChannels[i], self.tcurr, abs_tol=1e-9):
                self.tnextChannels[i] = float('inf')
                self.channels[i] = 0
                self.servedCount += 1

                if self.nextElement is not None:
                    self.nextElement.inAct()

                self.isResting[i] = 1
                self.restTimes[i] = self.tcurr + 5.0

        self.checkAndStart()

    def doStatistics(self, delta):
        self.meanBusyArea += sum(self.channels) * delta
        for i in range(self.channelCount):
            if self.channels[i] == 0 and self.isResting[i] == 0:
                self.areaIdle[i] += delta
            else:
                self.areaBusy[i] += delta


class Create:
    def __init__(self, name):
        self.name = name
        self.tnext = 0.0
        self.tcurr = 0.0
        self.delayMean = 0.0
        self.delayDev = 0.0
        self.distribution = ""
        self.nextElement = None

    def setDelay(self, mean, dev, dist):
        self.delayMean = mean
        self.delayDev = dev
        self.distribution = dist

    def getDelay(self):
        return FunRand.Erlang(int(self.delayMean), self.delayDev)

    def getTnext(self):
        return self.tnext

    def setTcurr(self, t):
        self.tcurr = t

    def outAct(self):
        self.tnext = self.tcurr + self.getDelay()
        if self.nextElement is not None:
            self.nextElement.inAct()

    def doStatistics(self, delta):
        pass


class QueueEarth(Process):
    def __init__(self, name):
        super().__init__(name, 0, float('inf'))
        self.loader = None

    def inAct(self):
        self.queue += 1
        if self.loader:
            self.loader.checkAndStart()


class QueueTrucks(Process):
    def __init__(self, name, numTrucks=4):
        super().__init__(name, 0, numTrucks)
        self.queue = numTrucks
        self.loader = None

    def inAct(self):
        self.queue += 1
        if self.loader:
            self.loader.checkAndStart()


class Dispose:
    def __init__(self, name, target):
        self.name = name
        self.tnext = float('inf')
        self.tcurr = 0.0
        self.returnTarget = target

    def getTnext(self): return self.tnext
    def setTcurr(self, t): self.tcurr = t
    def inAct(self): self.returnTarget.inAct()
    def outAct(self): pass
    def doStatistics(self, delta): pass


def run_simulation(n_bulldozers=1, n_trucks=4, n_loaders=2, time=100000.0, seed=42):
    random.seed(seed)

    lambd_eff = 0.25 * n_bulldozers
    g1 = Create("G1 (Бульдозер)")
    g1.setDelay(2.0, lambd_eff, "erlang")
    g1.tnext = g1.getDelay()

    q1 = QueueEarth("Q1 (Черга землі)")
    q2 = QueueTrucks("Q2 (Черга самоскидів)", numTrucks=n_trucks)
    loaders = LoaderProcess("D (Навантажувачі)", q1, q2, numLoaders=n_loaders)
    q1.loader = loaders
    q2.loader = loaders

    c1 = Process("C1 (Рух завантажених)", n_trucks, float('inf'))
    c1.setDelay(22.0, 10.0, "norm")
    c2 = Process("C2 (Розвантаження)", n_trucks, float('inf'))
    c2.setDelay(2.0, 8.0, "unif")
    c3 = Process("C3 (Повернення порожніх)", n_trucks, float('inf'))
    c3.setDelay(18.0, 10.0, "norm")
    endCycle = Dispose("Повернення в Q2", q2)

    g1.nextElement = q1
    loaders.nextElement = c1
    c1.nextElement = c2
    c2.nextElement = c3
    c3.nextElement = endCycle

    elements = [g1, q1, q2, loaders, c1, c2, c3, endCycle]
    lastEventTime = 0.0
    tcurr = 0.0

    while tcurr < time:
        tnext = min(e.getTnext() for e in elements)

        if tnext > time:
            delta = time - lastEventTime
            for e in elements:
                e.doStatistics(delta)
            break

        delta = tnext - lastEventTime
        for e in elements:
            e.doStatistics(delta)

        lastEventTime = tnext
        tcurr = tnext

        for e in elements:
            e.setTcurr(tcurr)

        active_elements = [e for e in elements if math.isclose(e.getTnext(), tcurr, abs_tol=1e-9)]
        active_elements.sort(key=lambda x: 1 if isinstance(x, Create) else 0)

        for e in active_elements:
            e.outAct()

    mean_q_earth = q1.meanQueueArea / time
    mean_q_trucks = q2.meanQueueArea / time
    mean_q_smo = (c1.meanQueueArea + c2.meanQueueArea + c3.meanQueueArea) / time

    loader_stats = []
    for i in range(n_loaders):
        k_idle = loaders.areaIdle[i] / time
        k_busy = loaders.areaBusy[i] / time
        loader_stats.append((k_idle, k_busy))

    return {
        "lambd": lambd_eff,
        "q_earth": mean_q_earth,
        "q_trucks": mean_q_trucks,
        "q_smo": mean_q_smo,
        "loaders": loader_stats,
        "throughput_per_hour": loaders.servedCount / (time / 60.0)
    }


if __name__ == "__main__":
    scenarios = [
        ("Конфігурація за умовою завдання", 1, 4, 2),
        ("Дефіцит самоскидів (нижня межа)", 1, 2, 2),
        ("Надлишок самоскидів (розрахункова норма)", 1, 7, 2),
        ("Один навантажувач (M = 14 хв)", 1, 4, 1),
        ("Два бульдозери (подвоєний потік землі)", 2, 4, 2),
        ("Два бульдозери та 7 самоскидів", 2, 7, 2),
        ("Збалансована масштабована (2 бульд., 7 сам., 3 нав.)", 2, 7, 3),
    ]

    print("=" * 138)
    print(f"{'Режим перевірки':<50} | {'Nбульд (λ)':<11} | {'Nсам':<4} | {'Nнав':<4} | "
          f"{'Q землі':<8} | {'Q сам.':<6} | {'Q СМО':<5} | {'Навант. 1':<11} | {'Навант. 2':<11} | {'Авто/год':<8}")
    print("=" * 138)

    for title, nb, nt, nl in scenarios:
        res = run_simulation(nb, nt, nl, time=100000.0, seed=42)
        l1_str = f"{res['loaders'][0][0]:.2f} / {res['loaders'][0][1]:.2f}"
        l2_str = f"{res['loaders'][1][0]:.2f} / {res['loaders'][1][1]:.2f}" if nl >= 2 else "     —     "
        b_str = f"{nb} ({res['lambd']:.2f})"
        print(f"{title:<50} | {b_str:<11} | {nt:<4d} | {nl:<4d} | "
              f"{res['q_earth']:<8.2f} | {res['q_trucks']:<6.2f} | {res['q_smo']:<5.2f} | "
              f"{l1_str:<11} | {l2_str:<11} | {res['throughput_per_hour']:<8.2f}")
    print("=" * 138)
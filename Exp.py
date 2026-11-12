import random
import math
import numpy as np
import matplotlib.pyplot as plt


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


def run_single_experiment(n_bulldozers=1, n_trucks=4, n_loaders=2,
                          time_total=18000.0, t_warmup=3000.0, seed=42,
                          loader_delays=None, record_trajectory=False, step_record=100.0):
    random.seed(seed)
    lambd_eff = 0.25 * n_bulldozers

    g1 = Create("G1 (Бульдозер)")
    g1.setDelay(2.0, lambd_eff, "erlang")
    g1.tnext = g1.getDelay()

    q1 = QueueEarth("Q1 (Черга землі)")
    q2 = QueueTrucks("Q2 (Черга самоскидів)", numTrucks=n_trucks)
    loaders = LoaderProcess("D (Навантажувачі)", q1, q2, numLoaders=n_loaders)
    if loader_delays is not None:
        loaders.loaderDelays = list(loader_delays)

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
    warmup_reset_done = (t_warmup <= 0.0)

    history_q_earth = []
    history_q_trucks = []
    max_q_earth = 0
    max_q_trucks = n_trucks

    traj_t = []
    traj_q_earth = []
    traj_q_trucks = []
    next_record_t = step_record

    while tcurr < time_total:
        tnext = min(e.getTnext() for e in elements)

        if not warmup_reset_done and tnext >= t_warmup:
            delta_warm = t_warmup - lastEventTime
            for e in elements:
                e.doStatistics(delta_warm)
            for e in elements:
                if hasattr(e, 'meanQueueArea'):
                    e.meanQueueArea = 0.0
                    e.meanBusyArea = 0.0
                    e.servedCount = 0
            loaders.areaIdle = [0.0] * n_loaders
            loaders.areaBusy = [0.0] * n_loaders
            history_q_earth.clear()
            history_q_trucks.clear()
            max_q_earth = q1.queue
            max_q_trucks = q2.queue
            lastEventTime = t_warmup
            warmup_reset_done = True

        if tnext > time_total:
            delta = time_total - lastEventTime
            for e in elements:
                e.doStatistics(delta)
            if warmup_reset_done and delta > 0:
                history_q_earth.append((q1.queue, delta))
                history_q_trucks.append((q2.queue, delta))
            break

        delta = tnext - lastEventTime
        for e in elements:
            e.doStatistics(delta)

        if warmup_reset_done and delta > 0:
            history_q_earth.append((q1.queue, delta))
            history_q_trucks.append((q2.queue, delta))
            if q1.queue > max_q_earth:
                max_q_earth = q1.queue
            if q2.queue > max_q_trucks:
                max_q_trucks = q2.queue

        lastEventTime = tnext
        tcurr = tnext

        if record_trajectory and tcurr >= next_record_t:
            traj_t.append(tcurr)
            traj_q_earth.append(q1.meanQueueArea / (tcurr if not warmup_reset_done else tcurr - t_warmup))
            traj_q_trucks.append(q2.meanQueueArea / (tcurr if not warmup_reset_done else tcurr - t_warmup))
            next_record_t += step_record

        for e in elements:
            e.setTcurr(tcurr)

        active_elements = [e for e in elements if math.isclose(e.getTnext(), tcurr, abs_tol=1e-9)]
        active_elements.sort(key=lambda x: 1 if isinstance(x, Create) else 0)

        for e in active_elements:
            e.outAct()

    obs_time = time_total - t_warmup if time_total > t_warmup else time_total
    mean_q_earth = q1.meanQueueArea / obs_time
    mean_q_trucks = q2.meanQueueArea / obs_time

    var_earth = sum(((val - mean_q_earth) ** 2) * dt for val, dt in history_q_earth) / obs_time
    var_trucks = sum(((val - mean_q_trucks) ** 2) * dt for val, dt in history_q_trucks) / obs_time

    idle_rates = [loaders.areaIdle[i] / obs_time for i in range(n_loaders)]
    busy_rates = [loaders.areaBusy[i] / obs_time for i in range(n_loaders)]
    mean_idle = sum(idle_rates) / n_loaders
    mean_busy = sum(busy_rates) / n_loaders
    throughput_hr = loaders.servedCount / (obs_time / 60.0)

    actual_earth_rate = (loaders.servedCount * 2.0) / obs_time if loaders.servedCount > 0 else 0.25
    actual_truck_rate = loaders.servedCount / obs_time if loaders.servedCount > 0 else 0.0625
    wait_earth = mean_q_earth / actual_earth_rate
    wait_trucks = mean_q_trucks / actual_truck_rate

    return {
        "q_earth": mean_q_earth,
        "std_earth_within": math.sqrt(var_earth),
        "max_q_earth": max_q_earth,
        "wait_earth": wait_earth,
        "q_trucks": mean_q_trucks,
        "std_trucks_within": math.sqrt(var_trucks),
        "max_q_trucks": max_q_trucks,
        "wait_trucks": wait_trucks,
        "idle_rates": idle_rates,
        "busy_rates": busy_rates,
        "mean_idle": mean_idle,
        "mean_busy": mean_busy,
        "throughput": throughput_hr,
        "traj_t": traj_t,
        "traj_q_earth": traj_q_earth,
        "traj_q_trucks": traj_q_trucks
    }


def calculate_p_chebyshev(n_bulldozers=1, n_trucks=4, n_loaders=2,
                          t_warmup=3000.0, t_stat=15000.0,
                          epsilon_percent=0.05, beta=0.95):
    print("\n--- ПРОБНІ 5 ПРОГОНІВ ТА РОЗРАХУНОК ЗА ЧЕБИШЕВИМ ---")

    time_total = t_warmup + t_stat
    print(f"Час моделювання: T_пер ({t_warmup}) + T_стац ({t_stat}) = T_загал ({time_total}) хв")

    pilot_runs_earth = []

    for r in range(5):
        res = run_single_experiment(
            n_bulldozers=n_bulldozers, n_trucks=n_trucks, n_loaders=n_loaders,
            time_total=time_total, t_warmup=t_warmup, seed=777 + r
        )
        pilot_runs_earth.append(res["q_earth"])
        print(f"  Пробний прогін {r + 1}: L_землі = {res['q_earth']:.2f}")

    mean_earth = np.mean(pilot_runs_earth)
    variance_earth = np.var(pilot_runs_earth, ddof=1)

    epsilon = epsilon_percent * mean_earth

    denominator = (epsilon ** 2) * (1.0 - beta)
    p_calc = variance_earth / denominator
    p_required = math.ceil(p_calc)

    if p_required < 5:
        p_required = 5

    print(f"\nСереднє значення: {mean_earth:.4f}")
    print(f"Дисперсія: {variance_earth:.4f}")
    print(f"Задана точність: {epsilon:.4f} (це {epsilon_percent * 100}% від {mean_earth:.2f})")
    print(f"Довірча ймовірність: {beta}")
    print(f"Розрахована кількість прогонів: {p_required}")

    return p_required, time_total


def run_replications(n_bulldozers=1, n_trucks=4, n_loaders=2,
                     p_runs=20, time_total=18000.0, t_warmup=3000.0, base_seed=100, loader_delays=None):
    runs = []
    for r in range(p_runs):
        res = run_single_experiment(
            n_bulldozers=n_bulldozers, n_trucks=n_trucks, n_loaders=n_loaders,
            time_total=time_total, t_warmup=t_warmup, seed=base_seed + r * 17,
            loader_delays=loader_delays
        )
        runs.append(res)

    def stats_for(key):
        vals = [x[key] for x in runs]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if p_runs > 1 else 0.0
        ci_95 = 1.96 * std_val / math.sqrt(p_runs) if p_runs > 0 else 0.0
        return mean_val, std_val, ci_95, vals

    q_earth_m, q_earth_s, q_earth_ci, q_earth_vals = stats_for("q_earth")
    q_trucks_m, q_trucks_s, q_trucks_ci, q_trucks_vals = stats_for("q_trucks")
    idle_m, idle_s, idle_ci, idle_vals = stats_for("mean_idle")
    busy_m, busy_s, busy_ci, _ = stats_for("mean_busy")
    tp_m, tp_s, tp_ci, tp_vals = stats_for("throughput")
    w_earth_m, _, _, _ = stats_for("wait_earth")
    w_trucks_m, _, _, _ = stats_for("wait_trucks")

    return {
        "q_earth_mean": q_earth_m, "q_earth_std": q_earth_s, "q_earth_ci": q_earth_ci, "q_earth_vals": q_earth_vals,
        "q_trucks_mean": q_trucks_m, "q_trucks_std": q_trucks_s, "q_trucks_ci": q_trucks_ci, "q_trucks_vals": q_trucks_vals,
        "idle_mean": idle_m, "idle_std": idle_s, "idle_ci": idle_ci, "idle_vals": idle_vals,
        "busy_mean": busy_m, "busy_std": busy_s,
        "tp_mean": tp_m, "tp_std": tp_s,
        "wait_earth_mean": w_earth_m, "wait_trucks_mean": w_trucks_m,
        "max_q_earth": max(x["max_q_earth"] for x in runs),
        "max_q_trucks": max(x["max_q_trucks"] for x in runs)
    }


def experiment_1_transient_period():
    print("\n=== ЕТАП 1. ДОСЛІДЖЕННЯ ТРИВАЛОСТІ ПЕРЕХІДНОГО ПЕРІОДУ (5 РЕАЛІЗАЦІЙ) ===")
    seeds = [11, 22, 33, 44, 55]
    trajectories_earth = []
    trajectories_trucks = []
    t_grid = None

    plt.figure(figsize=(11, 4.8))
    plt.subplot(1, 2, 1)
    for idx, s in enumerate(seeds):
        res = run_single_experiment(1, 4, 2, time_total=15000.0, t_warmup=0.0,
                                    seed=s, record_trajectory=True, step_record=50.0)
        t_grid = res["traj_t"]
        trajectories_earth.append(res["traj_q_earth"])
        trajectories_trucks.append(res["traj_q_trucks"])
        plt.plot(t_grid, res["traj_q_earth"], label=f"Прогін {idx + 1} (seed={s})", linewidth=1.2)

    plt.axvline(x=3000, color='red', linestyle='--', label="T_пер = 3000 хв")
    plt.title("а) Стабілізація середньої черги землі L_землі(t)")
    plt.xlabel("Модельний час t, хв")
    plt.ylabel("Поточна середня черга землі, куп")
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for idx, s in enumerate(seeds):
        plt.plot(t_grid, trajectories_trucks[idx], label=f"Прогін {idx + 1} (seed={s})", linewidth=1.2)
    plt.axvline(x=3000, color='red', linestyle='--', label="T_пер = 3000 хв")
    plt.title("б) Стабілізація середньої черги самоскидів L_сам(t)")
    plt.xlabel("Модельний час t, хв")
    plt.ylabel("Поточна середня черга самоскидів, авто")
    plt.grid(True, alpha=0.4)
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("fig_4_1_transient.png", dpi=200)
    plt.close()

    print(f"{'Час t, хв':<10} | {'Макс. різниця |y_i(t)-y_j(t)| (земля)':<38} | {'Макс. різниця |y_i(t)-y_j(t)| (самоскиди)':<40}")
    print("-" * 95)

    for check_t in [500, 1000, 2000, 3000, 5000, 10000, 14900]:
        idx = min(range(len(t_grid)), key=lambda i: abs(t_grid[i] - check_t))

        vals_e = [tr[idx] for tr in trajectories_earth]
        vals_tr = [tr[idx] for tr in trajectories_trucks]
        diff_e = max(vals_e) - min(vals_e)
        diff_tr = max(vals_tr) - min(vals_tr)
        print(f"{check_t:<10} | {diff_e:<38.4f} | {diff_tr:<40.4f}")


def experiment_2_baseline_and_anova():
    print("\n=== ЕТАП 2. БАЗОВІ СТАТИСТИЧНІ ХАРАКТЕРИСТИКИ ТА ДИСПЕРСІЙНИЙ АНАЛІЗ (ANOVA) ===")

    t_warmup = 3000.0
    t_stat = 15000.0

    p_calculated, time_total = calculate_p_chebyshev(
        n_bulldozers=1, n_trucks=4, n_loaders=2,
        t_warmup=t_warmup, t_stat=t_stat,
        epsilon_percent=0.05, beta=0.95
    )

    print(f"\n=== ЗАПУСК ОСНОВНОЇ СЕРІЇ З {p_calculated} ПРОГОНІВ ===")

    base_stats = run_replications(1, 4, 2, p_runs=p_calculated, time_total=time_total, t_warmup=t_warmup)

    print(f"Базова система (Nбульд=1, Nсам=4, Nнав=2, Tзагал={time_total} хв, p={p_calculated} замірів):")
    print(f" Середня черга землі L_землі: {base_stats['q_earth_mean']:.2f} ± {base_stats['q_earth_ci']:.2f} куп "
          f"(СКВ міжпрогонне σ = {base_stats['q_earth_std']:.2f}, макс = {base_stats['max_q_earth']} куп, сер. час очік. = {base_stats['wait_earth_mean']:.2f} хв)")
    print(f" Середня черга самоскидів L_сам: {base_stats['q_trucks_mean']:.2f} ± {base_stats['q_trucks_ci']:.2f} авто "
          f"(СКВ міжпрогонне σ = {base_stats['q_trucks_std']:.2f}, макс = {base_stats['max_q_trucks']} авто, сер. час очік. = {base_stats['wait_trucks_mean']:.2f} хв)")
    print(f" Середній простій навантажувачів: {base_stats['idle_mean']:.3f} (СКВ σ = {base_stats['idle_std']:.3f}), "
          f"коеф. використання = {base_stats['busy_mean']:.3f}")
    print(f" Продуктивність системи: {base_stats['tp_mean']:.2f} ± {base_stats['tp_std']:.2f} авто/год")

    levels_trucks = [3, 4, 5, 6, 7]
    q = len(levels_trucks)
    samples_earth = []
    samples_trucks = []

    for nt in levels_trucks:
        st = run_replications(1, nt, 2, p_runs=p_calculated, time_total=time_total, t_warmup=t_warmup)
        samples_earth.append(st["q_earth_vals"])
        samples_trucks.append(st["q_trucks_vals"])

    def calc_anova(samples):
        means_j = [np.mean(s) for s in samples]
        grand_mean = np.mean(means_j)
        ss_f = p_calculated * sum((mj - grand_mean) ** 2 for mj in means_j)
        ss_r = sum(sum((y_js - means_j[j]) ** 2 for y_js in samples[j]) for j in range(q))
        df_doc = ss_f / (q * p_calculated - 1)
        dr_doc = ss_r / (q * (p_calculated - 1))
        ms_f = ss_f / (q - 1)
        f_ratio = ms_f / dr_doc
        return df_doc, dr_doc, f_ratio

    df_e, dr_e, f_e = calc_anova(samples_earth)
    df_t, dr_t, f_t = calc_anova(samples_trucks)
    print(f"\nДисперсійний аналіз (ANOVA) впливу фактора N_сам ∈ {{3, 4, 5, 6, 7}} (q={q}, p={p_calculated}):")
    print(f" Відгук L_землі: D_f = {df_e:.4f}, D_r = {dr_e:.4f}, Критерій Фішера F = {f_e:.2f} >> F_табл(4, {q * (p_calculated - 1)})")
    print(f" Відгук L_сам: D_f = {df_t:.4f}, D_r = {dr_t:.4f}, Критерій Фішера F = {f_t:.2f} >> F_табл(4, {q * (p_calculated - 1)})")

    return p_calculated, time_total, t_warmup


def experiment_3_regression_analysis(p_calculated, time_total, t_warmup):
    print("\n=== ЕТАП 3. РЕГРЕСІЙНИЙ АНАЛІЗ ЗА ФОРМУЛАМИ (3.6)-(3.9) ===")

    X_nat = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    X0 = (7.0 + 3.0) / 2.0
    delta = (7.0 - 3.0) / 2.0
    x_norm = (X_nat - X0) / delta

    y_earth = []
    y_trucks = []

    for nt in X_nat:
        st = run_replications(1, int(nt), 2, p_runs=p_calculated, time_total=time_total, t_warmup=t_warmup)
        y_earth.append(st["q_earth_mean"])
        y_trucks.append(st["q_trucks_mean"])

    Y_e = np.array(y_earth)
    Y_t = np.array(y_trucks)

    X_mat = np.column_stack([np.ones_like(x_norm), x_norm, x_norm ** 2])
    b_earth = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ Y_e)
    b_trucks = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ Y_t)

    y_e_pred = X_mat @ b_earth
    y_t_pred = X_mat @ b_trucks

    r_earth = math.sqrt(1.0 - np.sum((Y_e - y_e_pred) ** 2) / np.sum((Y_e - np.mean(Y_e)) ** 2))
    r_trucks = math.sqrt(1.0 - np.sum((Y_t - y_t_pred) ** 2) / np.sum((Y_t - np.mean(Y_t)) ** 2))

    print(f"Рівняння регресії для L_землі: y = {b_earth[0]:.3f} {b_earth[1]:+.3f}*x1 {b_earth[2]:+.3f}*x1^2 | Індекс кореляції R = {r_earth:.4f}")
    print(f"Рівняння регресії для L_сам: y = {b_trucks[0]:.3f} {b_trucks[1]:+.3f}*x1 {b_trucks[2]:+.3f}*x1^2 | Індекс кореляції R = {r_trucks:.4f}")

    x_dense = np.linspace(-1, 1, 100)
    X_dense_nat = X0 + delta * x_dense
    y_e_curve = b_earth[0] + b_earth[1] * x_dense + b_earth[2] * (x_dense ** 2)
    y_t_curve = b_trucks[0] + b_trucks[1] * x_dense + b_trucks[2] * (x_dense ** 2)

    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.scatter(X_nat, Y_e, color='darkblue', s=50, label=f"Експериментальні точки (p={p_calculated})")
    plt.plot(X_dense_nat, y_e_curve, color='crimson', linestyle='-', label=f"Регресія (R = {r_earth:.4f})")
    plt.title("а) Регресійна модель черги землі L_землі(N_сам)")
    plt.xlabel("Кількість самоскидів N_сам, шт.")
    plt.ylabel("Середня черга землі, куп")
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(X_nat, Y_t, color='darkgreen', s=50, label=f"Експериментальні точки (p={p_calculated})")
    plt.plot(X_dense_nat, y_t_curve, color='darkorange', linestyle='-', label=f"Регресія (R = {r_trucks:.4f})")
    plt.title("б) Регресійна модель черги самоскидів L_сам(N_сам)")
    plt.xlabel("Кількість самоскидів N_сам, шт.")
    plt.ylabel("Середня черга самоскидів, авто")
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig_4_2_regression.png", dpi=200)
    plt.close()


def experiment_4_optimization_search(p_calculated, time_total, t_warmup):
    print("\n=== ЕТАП 4. СЕРІЯ ЕКСПЕРИМЕНТІВ ДЛЯ ПОШУКУ ОПТИМАЛЬНИХ ПАРАМЕТРІВ (ЗМЕНШЕННЯ ЧЕРГ) ===")
    print("-" * 118)
    print(f"{'Nбульд':<6} | {'Nнав':<4} | {'Nсам':<4} | {'L_землі, куп':<16} | {'L_сам, авто':<15} | "
          f"{'Простій нав.':<13} | {'Використ. нав.':<14} | {'Авто/год':<10} | {'Режим системи'}")
    print("-" * 118)

    configs = [
        # 1 бульдозер (базовий вхідний потік 3.75 партій/год)
        (1, 1, 4), (1, 1, 5),
        (1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 6), (1, 2, 7),
        (1, 3, 4), (1, 3, 5),
        # 2 бульдозери (масштабований потік 7.50 партій/год)
        (2, 2, 7), (2, 2, 8), (2, 2, 9),
        (2, 3, 7), (2, 3, 8), (2, 3, 9), (2, 3, 10),
        (2, 4, 8), (2, 4, 9), (2, 4, 10)
    ]

    results_1b = []
    results_2b = []

    for nb, nl, nt in configs:
        st = run_replications(nb, nt, nl, p_runs=p_calculated, time_total=time_total, t_warmup=t_warmup)
        target_tp = 3.75 * nb

        is_stable = (st["q_earth_mean"] < 50.0) and (st["tp_mean"] >= 0.96 * target_tp)

        status = "Стаціонарний (оптимальна зона)" if is_stable else "Нестаціонарний (ріст черги Q1)"
        if is_stable and st["q_trucks_mean"] > 1.8:
            status = "Стаціонарний (надлишок авто)"

        print(f"{nb:<6d} | {nl:<4d} | {nt:<4d} | {st['q_earth_mean']:6.2f} ± {st['q_earth_ci']:4.2f} | "
              f"{st['q_trucks_mean']:5.2f} ± {st['q_trucks_ci']:4.2f} | {st['idle_mean']:<13.2f} | "
              f"{st['busy_mean']:<14.2f} | {st['tp_mean']:<10.2f} | {status}")

        if nb == 1 and nl == 2:
            results_1b.append((nt, st["q_earth_mean"], st["q_trucks_mean"], st["idle_mean"]))
        if nb == 2 and nl == 3:
            results_2b.append((nt, st["q_earth_mean"], st["q_trucks_mean"], st["tp_mean"]))

    print("-" * 118)

    st_mod = run_replications(1, 5, 2, p_runs=p_calculated, time_total=time_total, t_warmup=t_warmup,
                              loader_delays=[12.0, 12.0])
    print(f"Модернізований режим (Nбульд=1, Nнав=2 [M1=M2=12 хв], Nсам=5): "
          f"L_землі = {st_mod['q_earth_mean']:.2f} куп, L_сам = {st_mod['q_trucks_mean']:.2f} авто, "
          f"Продуктивність = {st_mod['tp_mean']:.2f} авто/год")

    nt_vals = [x[0] for x in results_2b]
    qe_vals = [x[1] for x in results_2b]
    qt_vals = [x[2] for x in results_2b]
    tp_vals = [x[3] for x in results_2b]

    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(nt_vals, qe_vals, marker='o', color='firebrick', linewidth=2, label="Черга землі L_землі")
    plt.title("а) Зменшення черги землі при N_бульд=2, N_нав=3")
    plt.xlabel("Кількість самоскидів N_сам, шт.")
    plt.ylabel("Середня черга землі, куп")
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(nt_vals, qt_vals, marker='s', color='navy', linewidth=2, label="Черга самоскидів L_сам")
    plt.plot(nt_vals, tp_vals, marker='^', color='seagreen', linewidth=2, label="Продуктивність, авто/год")
    plt.title("б) Черга самоскидів та продуктивність (N_бульд=2, N_нав=3)")
    plt.xlabel("Кількість самоскидів N_сам, шт.")
    plt.ylabel("Значення показника")
    plt.grid(True, alpha=0.4)
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig_4_3_optimization_2b.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    experiment_1_transient_period()

    p_calculated, time_total, t_warmup = experiment_2_baseline_and_anova()
    experiment_3_regression_analysis(p_calculated, time_total, t_warmup)
    experiment_4_optimization_search(p_calculated, time_total, t_warmup)
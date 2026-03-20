"""
QAOA - Quantum Approximate Optimization Algorithm
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize as scipy_minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.utils import algorithm_globals

SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

CSV_DRAWN = "/data/loto7hh_4582_k22.csv"
CSV_ALL   = "/data/kombinacijeH_39C7.csv"

MIN_VAL = [1, 2, 3, 4, 5, 6, 7]
MAX_VAL = [33, 34, 35, 36, 37, 38, 39]
NUM_QUBITS = 5
QAOA_DEPTH = 4
MAXITER = 250


def load_draws():
    df = pd.read_csv(CSV_DRAWN)
    return df.values


def build_empirical(draws, pos):
    n_states = 1 << NUM_QUBITS
    freq = np.zeros(n_states)
    for row in draws:
        v = int(row[pos]) - MIN_VAL[pos]
        if v >= n_states:
            v = v % n_states
        freq[v] += 1
    return freq / freq.sum()


def build_cost_hamiltonian(emp):
    n_states = 1 << NUM_QUBITS
    C = np.zeros(n_states)
    for i in range(n_states):
        C[i] = -np.log(max(emp[i], 1e-10))
    C = C - C.min()
    if C.max() > 0:
        C = C / C.max() * np.pi
    return C


def qaoa_circuit(gamma, beta, C):
    qc = QuantumCircuit(NUM_QUBITS)

    for i in range(NUM_QUBITS):
        qc.h(i)

    for p in range(QAOA_DEPTH):
        for i in range(NUM_QUBITS):
            bit_contrib = np.zeros(NUM_QUBITS)
            for s in range(1 << NUM_QUBITS):
                for q in range(NUM_QUBITS):
                    if (s >> q) & 1:
                        bit_contrib[q] += C[s]
            bit_contrib /= (1 << (NUM_QUBITS - 1))
            qc.rz(2 * gamma[p] * bit_contrib[i], i)

        for i in range(NUM_QUBITS - 1):
            pair_contrib = 0.0
            for s in range(1 << NUM_QUBITS):
                if ((s >> i) & 1) and ((s >> (i + 1)) & 1):
                    pair_contrib += C[s]
            pair_contrib /= (1 << (NUM_QUBITS - 2))
            qc.rzz(gamma[p] * pair_contrib * 0.1, i, i + 1)

        for i in range(NUM_QUBITS):
            qc.rx(2 * beta[p], i)

    return qc


def train_qaoa(emp):
    C = build_cost_hamiltonian(emp)
    n_params = 2 * QAOA_DEPTH
    params0 = np.random.uniform(0, np.pi, n_params)

    def cost(params):
        gamma = params[:QAOA_DEPTH]
        beta = params[QAOA_DEPTH:]
        qc = qaoa_circuit(gamma, beta, C)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities()
        exp_cost = np.dot(probs, C)
        return float(exp_cost)

    res = scipy_minimize(cost, params0, method='COBYLA',
                         options={'maxiter': MAXITER, 'rhobeg': 0.5})

    gamma = res.x[:QAOA_DEPTH]
    beta = res.x[QAOA_DEPTH:]
    qc = qaoa_circuit(gamma, beta, C)
    sv = Statevector.from_instruction(qc)
    return sv.probabilities(), res.fun


def greedy_combo(dists):
    combo = []
    used = set()
    for pos in range(7):
        ranked = sorted(enumerate(dists[pos]),
                        key=lambda x: x[1], reverse=True)
        for mv, score in ranked:
            actual = int(mv) + MIN_VAL[pos]
            if actual > MAX_VAL[pos]:
                continue
            if actual in used:
                continue
            if combo and actual <= combo[-1]:
                continue
            combo.append(actual)
            used.add(actual)
            break
    return combo


def main():
    draws = load_draws()
    print(f"Ucitano izvucenih kombinacija: {len(draws)}")

    df_all_head = pd.read_csv(CSV_ALL, nrows=3)
    print(f"Graf svih kombinacija: {CSV_ALL}")
    print(f"  Primer: {df_all_head.values[0].tolist()} ... "
          f"{df_all_head.values[-1].tolist()}")

    print(f"\n--- QAOA ({NUM_QUBITS}q, depth={QAOA_DEPTH}, "
          f"COBYLA {MAXITER} iter) ---")
    print(f"  Parametara: {2 * QAOA_DEPTH} (gamma + beta)")

    dists = []
    for pos in range(7):
        print(f"  Poz {pos+1}...", end=" ", flush=True)
        emp = build_empirical(draws, pos)
        born, obj = train_qaoa(emp)
        dists.append(born)

        top_idx = np.argsort(born)[::-1][:3]
        info = " | ".join(
            f"{i + MIN_VAL[pos]}:{born[i]:.3f}" for i in top_idx)
        print(f"obj={obj:.4f}  top: {info}")

    combo = greedy_combo(dists)

    print(f"\n{'='*50}")
    print(f"Predikcija (QAOA, deterministicki, seed={SEED}):")
    print(combo)
    print(f"{'='*50}")


if __name__ == "__main__":
    main()


"""
Ucitano izvucenih kombinacija: 4582
Graf svih kombinacija: /data/kombinacijeH_39C7.csv
  Primer: [1, 2, 3, 4, 5, 6, 7] ... [1, 2, 3, 4, 5, 6, 9]

--- QAOA (5q, depth=4, COBYLA 250 iter) ---
  Parametara: 8 (gamma + beta)
  Poz 1... obj=0.1019  top: 4:0.262 | 2:0.237 | 3:0.194
  Poz 2... obj=0.0638  top: 9:0.144 | 5:0.125 | 7:0.099
  Poz 3... obj=0.1896  top: 11:0.484 | 19:0.110 | 23:0.086
  Poz 4... obj=0.2707  top: 25:0.148 | 14:0.133 | 17:0.092
  Poz 5... obj=0.0668  top: 29:0.660 | 21:0.223 | 33:0.041
  Poz 6... obj=0.0558  top: 29:0.816 | 37:0.071 | 28:0.055
  Poz 7... obj=0.0674  top: 7:0.916 | 23:0.075 | 11:0.003

==================================================
Predikcija (QAOA, deterministicki, seed=39):
[4, 9, x, y, z, 37, 38]
==================================================
"""


"""
QAOA - Quantum Approximate Optimization Algorithm
QAOA je kvantni algoritam za optimizaciju funkcija.
QAOA se sastoji od 5 qubita i 4 sloja Ry+CX+Rz rotacija.

Cost hamiltonijan iz podataka: C(s) = -log(frekvencija stanja s) - nizak "cost" za frekventna stanja
QAOA struktura (depth=4):
Problem sloj (gamma): Rz od cost doprinosa po qubitu + Rzz za parove (Ising interakcija)
Mixer sloj (beta): Rx na svim qubitima
4 naizmenicna sloja
Optimizacija: COBYLA minimizuje ocekivani cost = sum(probs * C) - gura verovatnocu ka stanjima sa visokom frekvencijom
Samo 8 parametara (4 gamma + 4 beta) - efikasan parametarski prostor
Standardni QAOA framework prilagodjen za kombinatornu optimizaciju nad distribucijom
Deterministicki, Statevector

QAOA je kvantni optimizator, ne regresor. 
Minimizuje cost funkciju umesto da fituje regresioni model. 
Koristi ga za nalazenje stanja sa najnizim "costom" (najvisim frekvencijama), 
ali to nije regresija.
"""

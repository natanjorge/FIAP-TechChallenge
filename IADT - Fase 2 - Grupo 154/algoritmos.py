import numpy as np
import random

def calcula_distancia(rota, pontos):
    distancia = 0
    for i in range(1, len(rota)):
        a = pontos[rota[i - 1]]
        b = pontos[rota[i]]
        distancia += np.linalg.norm(a - b)
    return distancia

def gera_populacao(n_pop, n_clientes):
    return [random.sample(range(n_clientes), n_clientes) for _ in range(n_pop)]

def seleciona_populacao(populacao, pontos, n_best):
    distancias = [(calcula_distancia(rota, pontos), rota) for rota in populacao]
    distancias.sort()
    return [rota for _, rota in distancias[:n_best]]

def crossover(rota1, rota2):
    a, b = sorted(random.sample(range(len(rota1)), 2))
    filho = [None] * len(rota1)
    filho[a:b] = rota1[a:b]
    ptr = b
    for gene in rota2[b:] + rota2[:b]:
        if gene not in filho:
            if ptr >= len(filho):
                ptr = 0
            filho[ptr] = gene
            ptr += 1
    return filho

def mutacao(rota, taxa=0.1):
    rota = rota.copy()
    if random.random() < taxa:
        a, b = random.sample(range(len(rota)), 2)
        rota[a], rota[b] = rota[b], rota[a]
    return rota

def algoritmo_genetico(pontos, n_pop=60, n_gen=180, taxa_mut=0.15, n_best=10):
    populacao = gera_populacao(n_pop, len(pontos))
    melhor_dist = float('inf')
    melhor_rota = None
    for gen in range(n_gen):
        selecionados = seleciona_populacao(populacao, pontos, n_best)
        nova_pop = selecionados.copy()
        while len(nova_pop) < n_pop:
            pais = random.sample(selecionados, 2)
            filho = crossover(pais[0], pais[1])
            filho = mutacao(filho, taxa_mut)
            nova_pop.append(filho)
        populacao = nova_pop
        dist_atual, rota_atual = calcula_distancia(selecionados[0], pontos), selecionados[0]
        if dist_atual < melhor_dist:
            melhor_dist = dist_atual
            melhor_rota = rota_atual
    return melhor_rota, melhor_dist

def nearest_neighbor(pontos):
    n = len(pontos)
    visitado = [False] * n
    ordem = [0]
    visitado[0] = True
    for _ in range(n - 1):
        ult = ordem[-1]
        menor_dist = float('inf')
        prox = -1
        for i in range(n):
            if not visitado[i]:
                dist = np.linalg.norm(pontos[ult] - pontos[i])
                if dist < menor_dist:
                    menor_dist = dist
                    prox = i
        ordem.append(prox)
        visitado[prox] = True
    return ordem

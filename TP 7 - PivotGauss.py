import numpy as np


# C = np.array([[2, 6, 1, 2], [1, 1, 0, 5], [5, 9, 2, 1], [5, 7, 3, 8], [1, 1, 5, 5], [1, 1, 0, 5]])
# D = np.array([1, 1, 2, 5, 1, 1])

A = np.array([[11, 8, 8, 13, 18, 10, 12, 14, 18, 3],
              [3, 13, 19, 2, 14, 15, 11, 10, 16, 5],
              [17, 4, 16, 2, 5, 14, 19, 13, 17, 7],
              [5, 10, 15, 6, 7, 17, 16, 12, 2, 5],
              [9, 14, 18, 15, 16, 2, 8, 10, 16, 6],
              [4, 6, 16, 17, 5, 13, 18, 16, 3, 14],
              [11, 15, 16, 3, 11, 0, 19, 8, 9, 8],
              [3, 5, 16, 8, 16, 0, 2, 18, 14, 6],
              [15, 1, 9, 5, 11, 1, 0, 6, 12, 15],
              [10, 11, 13, 18, 5, 17, 17, 2, 12, 3]], dtype=object)
B = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=object)


C = np.array([[2, 6, 1, 2], [1, 1, 0, 5], [5, 9, 2, 1]])
D = np.array([1, 1, 2])

E = np.array([[2, 6, 1, 0, 2], [1, 1, 0, 0, 5], [5, 9, 2, 0, 1]])


#
# Affichage & Simplifications
#


def AfficherSystem(A, B):
    shape = A.shape
    print('---------------------')
    for i in range(shape[0]):
        print('L' + str(i), end=' : ')
        for j in range(shape[1]):
            print(str(A[i, j]) + '*x' + str(j), end='')
            if j < shape[1] - 1:
                print(' + ', end='')
        print(' = ' + str(B[i]))
    print('---------------------')


def AffichageSol(Sol):
    n, p = Sol.shape

    for i in range(n - 1):
        # on affiche le nom de l'inconnue
        coef = Sol[i, 0]
        if coef > 0:
            print(str(coef) + "x_" + str(i) + " = ", end="")
        elif coef < 0:
            print(" - " + str(-coef) + "x_" + str(i) + " = ", end="")

        inconnueAjoute = False
        # on affiche les inconnues paramètres
        for j in range(1, p - 1):
            # le numéro de l'inconnue est sur la dernière ligne
            coef = Sol[i, j]
            if coef > 0:
                inconnueAjoute = True
                print(" + " + str(coef) + "x_" + str(Sol[n - 1, j]), end="")
            elif coef < 0:
                inconnueAjoute = True
                print(" - " + str(-coef) + "x_" + str(Sol[n - 1, j]), end="")

        # on affiche le coef du second membre
        coef = Sol[i, p - 1]
        if coef > 0:
            print(" + " + str(coef), end="")
        elif coef < 0:
            print(" - " + str(-coef), end="")
        elif not inconnueAjoute:
            print("0", end="")
        print()


def PGCD(a, b):
    """
    PGCD de a et b

    :param a:
    :param b:
    :return pgcd:
    """
    u = abs(a)
    v = abs(b)

    while v > 0:
        u, v = v, u % v

    return u


def PPCM(a, b):
    """
    PPCM de a et b

    :param a:
    :param b:
    :return ppcm:
    """

    return abs(a * b) // PGCD(a, b)


def SimplifierSys(A, B):
    """
    On simplifie chaque ligne par le PGCD de tous les coeficients de celle ci

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    """
    n, p = A.shape

    for i in range(n):
        pgcd = A[i, 0]

        # pour chaque colonne
        for j in range(p):
            pgcd = PGCD(pgcd, A[i, j])
        # pgcd avec le second membre
        pgcd = PGCD(pgcd, B[i])

        # si ils sont pas premier
        if pgcd > 1:

            # on simplifie
            for j in range(p):
                A[i, j] = A[i, j] // pgcd

            B[i] = B[i] // pgcd

    print('Simplification du système')

#
# OPERATIONS ELEMENTAIRES
#


def Dilatation(A, B, i, a):
    """
    Réalise l'opération élémentaire : L_i <- aL_i

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    :param i: la ligne
    :param a: le coeficient de dilation
    """
    if a != 0:
        shape = A.shape

        for k in range(shape[1]):
            A[i, k] *= a

        B[i] *= a
        print('L' + str(i) + ' <- ' + str(a) + '*L' + str(i))


def Transvection(A, B, i1, i2, a):
    """
    Réalise l'opération élémentaire : L_i <- L_i1 + aL_i2

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    :param i1: la première ligne
    :param i2: la seconde ligne
    :param a: le coeficient de dilation
    """
    if i1 != i2:
        shape = A.shape

        for k in range(shape[1]):
            A[i1, k] = A[i1, k] + a * A[i2, k]

        B[i1] += a * B[i2]
        print('L' + str(i1) + ' <- L' + str(i1) + ' + ' + str(a) + '*L' + str(i2))


def Transpostion(A, B, i1, i2):
    """
    Réalise l'opération élémentaire : L_i1 <-> L_i2

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    :param i1: la première ligne
    :param i2: la seconde ligne
    """
    shape = A.shape

    for k in range(shape[1]):
        A[i1, k], A[i2, k] = A[i2, k], A[i1, k]

    B[i1], B[i2] = B[i2], B[i1]
    print('L' + str(i1) + ' <-> L' + str(i2))


def IndicePivot(A, i, j):
    """
    Renvoie l'indice du premier pivot de la colonne j en partant de la ligne i

    :param A: le tableau associés aux inconnues
    :param i: ligne de départ
    :param j: colonne
    :return r, q: boolean (si le pivot est trouvé), int (la ligne du premier pivot)
    """
    n, p = A.shape
    q = i
    r = False

    while q < n and not r:
        if A[q, j] != 0:
            r = True
        else:
            q += 1

    return r, q

#
# Gausss
#


def GaussEntier(A, B):
    """
    Calcul un système échelonnée réduite

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    """
    n, p = A.shape
    # pour chaque colonne
    for j in range(p):
        # on cherche un pivot pour la colonne j en partant de la ligne j
        r, q = IndicePivot(A, j, j)

        # si on a un pivot
        if r:
            # on recup le coef du pivot
            coefPivot = A[j, q]

            # on echelonne la colonne
            # pour chaque ligne
            for i in range(n):
                # si c'est pas la ligne du pivot
                if i != q:
                    # on recup le coef de la ligne
                    coef = A[i, j]
                    if coef != 0:
                        ppcm = PPCM(coefPivot, coef)

                        # on dilate pour toujours garder des entiers
                        Dilatation(A, B, i, ppcm // coef)
                        # on annule le coefficient i, j
                        Transvection(A, B, i, j, -ppcm // coefPivot)

        SimplifierSys(A, B)

    AfficherSystem(A, B)


def GaussSol(A, B):
    """
    Prend un système échelonné reduit et sans equation de compatibilitée, et renvoie des informations sur le système

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    :return:
        boolean[] L1 : Si l'inconnue de colonne i est pivot ou pas
        int[] L2 : indice de l'inconnue de la colonne i (indice des pivot est séparé des inconnues paramètres)
        int[] L3 : indice de colonne des inconnues paramètres
        int r : le rang (nombre de pivot)

    """
    n, p = A.shape

    L1 = [False for x in range(p)]
    L2 = [-1 for x in range(p)]  # si ya des -1 en sortie c'est que ya un pb
    L3 = []  #
    r = 0

    incoP = 0
    incoA = 0

    # pour chaque ligne
    for i in range(n):

        # pour chaque colonne
        for j in range(p):
            # si la case i, j est la première case non nulle
            if A[i, j] != 0:
                # alors la colonne j est un pivot
                L1[j] = True
                # on ajoute son indice de pivot
                L2[j] = incoP
                incoP += 1
                # on augmente de le range
                r += 1

                break

    # pour chaque colonne
    for i in range(p):
        # si la colonne n'est pas un pivot
        if not L1[i]:
            # on ajoute son indice en tant qu'inconnue paramètre
            L2[i] = incoA
            incoA += 1
            # on ajoute à la liste des inconnues paramètres
            L3.append(i)

    return L1, L2, L3, r


def SolveEntier(A, B):
    """
    Prend un sytème. Echlonne et réduit le système. Vérifie les équations de compatibilité.
    Renvoie la(es) solution(s)

    :param A: le tableau associés aux inconnues
    :param B: le tableau associés au second membre
    :return:
    """
    A_r = A.copy()
    B_r = B.copy()
    # on echelonne reduit le système
    GaussEntier(A_r, B_r)

    n, p = A.shape
    S = None
    compatible = True

    if p < n:  # Si y a moins d'inconnue que de ligne, alors on verifie les equations de compatibilitées

        # si une des lignes entre p (inclu) et n (exclu) du second membre est non nul
        # alors le système n'est pas compatible
        for i in range(p, n):
            if B_r[i] != 0:
                compatible = False
                break

        if compatible:  # Si système compatible, on enleve les equations de compatibilitées
            A_r = A_r[0:p, :]
            B_r = B_r[0:p]

    if compatible:
        # on recupère les informations sur le système
        # echelonnée reduit et sans equation de compatibilitée
        L1, L2, L3, r = GaussSol(A_r, B_r)

        # on fait un tableau de taille
        # lign : r (rang = nombre de pivot) + 1 numero d'inconnue paramètre
        # colonne : 1 (pivot) + p - r (nombre d'inconnue paramètre) + 1 (second membre)
        S = np.zeros((r + 1, 2 + p - r), dtype=object)

        for i in range(r):  # ajout coef du second membre
            S[i, p - r + 1] = B_r[i]

        for j in range(p):  # ajoute coef des pivots et des inconnues paramètres
            # si c'est un pivot
            if L1[j]:
                S[j, 0] = A_r[j, j]
            else:
                # pour inconnue pivot
                for i in range(r):
                    # on ajoute le coef de l'inconnue paramètre
                    S[i, L2[j] + 1] = -A_r[i, L3[L2[j]]]

                # pour cette inconnues paramètre
                # on ajoute le numero de l'inconnue dans la dernière ligne sous la colonne
                S[r, L2[j] + 1] = L3[L2[j]]

    return S


AffichageSol(SolveEntier(A, B))
AfficherSystem(A, B)
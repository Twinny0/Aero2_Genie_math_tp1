# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 8:34:31 2021

@author: theod
"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt
#############Partie1###############

def ReductionGauss(Aaug):
    n, m = np.shape(Aaug)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            g = Aaug[j, i] / Aaug[i, i]
            Aaug[j, :] = Aaug[j, :] - g * Aaug[i, :]
    return Aaug


def ResolutionSystTriSup(Taug):
    n, m = np.shape(Taug)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        somme = 0
        for j in range(i, n):
            somme = somme + Taug[i, j] * x[j]
            x[i] = (Taug[i, n] - somme) / Taug[i, i]
    return x


def Gauss(A, B):
    Aaug = np.concatenate((A, B), axis=1)
    mat_x = ResolutionSystTriSup(ReductionGauss(Aaug))
    return mat_x


def fonction_matrice_A(x):
    A = np.zeros((x, x))
    n, m = np.shape(A)
    for i in range(0, n):
        for j in range(0, m):
            A[i, j] = float(random.randint(-x, x))
    return A


def fonction_matrice_B(x):
    B = np.zeros((x, 1))
    n, m = np.shape(B)
    for i in range(0, n):
        for j in range(0, m):
            B[i, j] = float(random.randint(-x, x))
    return B



def complexitegauss():
    complexite = []
    taille = []
    for i in range(25, 2525,25):
        comp = (2/3) * i ** 3
        complexite.append(comp)
        taille.append(i)
    plt.plot(np.array(taille), np.array(complexite), label = "Equation de la compléxité")
    plt.xlabel("Taille de matrice")
    plt.ylabel("Complexité")
    plt.title("Evolution du nombre de calcul en fonction de la dimension de la matrice")
    plt.legend()    
    plt.show()    



def Question4():
    t = 25
    matrice_a = np.zeros((1, 1))
    matrice_b = np.zeros((1, 1))   
    taille = []
    temps_cpu = []
    erreur = []
    td = time.time()
    while t <= 2500:
        AX = 0
        X = 0
        matrice_a = fonction_matrice_A(t)
        matrice_b = fonction_matrice_B(t)
            
        debut_cpu = time.process_time()
        X = Gauss(matrice_a, matrice_b)
        fin_cpu = time.process_time()
            
        delta_cpu = fin_cpu - debut_cpu
        
        taille.append(t)
        temps_cpu.append(delta_cpu)
            
        AX = np.dot(matrice_a, X)
        sous = AX - matrice_b
            
        n, m = np.shape(sous)
        diag = 0
        for i in range(n):
            diag += abs(sous[i, i])
        erreur.append(diag)
            
        print('TEMPS CPU Matrice de taille:', t, delta_cpu)
        
        t += 25
            
        print("DONE")
    tf = time.time()
    z = tf - td
    print(z)
    print("END")
    plt.plot(np.array(taille), np.array(temps_cpu), label = "Temps CPU")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Temps (s)")
    plt.title("Evolution du temps de calcul en fonction de la dimension de la matrice")
    plt.legend()
    
    plt.show()
    
    plt.plot(np.array(taille), np.array(erreur), label = "Erreur")
    plt.xlabel("Taille de matrice")
    plt.ylabel("Erreur")
    plt.title("Evolution de l'erreur en fonction de la dimension de la matrice")
    plt.legend()
    
    plt.show()
    

    
    
    
##############Partie2##################
  

          
b = np.array([[2 ,5, 6], [4, 11, 9], [-2, -8, 7]])

def DecompostionLU(A):
    n, m = np.shape(A)
    L = np.identity(n)
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            g = A[j, i] / A[i, i]
            L[j, i] = g
            A[j, :] = A[j, :] - g * A[i, :]
    return [A, L]


def ResolutionSystTriInf(Taug):
    n, m = np.shape(Taug)
    x = np.zeros(n)
    x[0] = Taug[0, m - 1] / Taug[0, 0]
    for i in range(1, n):
        x[i] = Taug[i, n]
        for j in range (0, i):
            x[i] = x[i] - Taug[i, j] * x[j]
        x[i] = x[i] / Taug[i, i]
    new_x = x.reshape(-1, 1)
    return new_x


def ResolutionLU(L, U, B):
    Laug = np.concatenate((L, B), axis = 1)
    Y = ResolutionSystTriInf(Laug)
    Uaug = np.concatenate((U, Y), axis = 1)
    mat = ResolutionSystTriSup(Uaug)
    result = mat.reshape(-1, 1)
    return result


def Question2(A, B):
    LowerUpper = DecompostionLU(A)
    resultat = ResolutionLU(LowerUpper[1], LowerUpper[0], B)
    return resultat

def TestQuestion2():
    t = 25
    taille = []
    temps_cpu = []
    erreur = []
    td = time.time()
    while t <= 400 :
        maa = fonction_matrice_A(t)
        mbb = fonction_matrice_B(t)

        debut = time.process_time()
            
        X = Question2(maa, mbb)
            
        fin = time.process_time()
        delta = fin - debut
        print('TEMPS CPU Matrice de taille:', t, delta)
            
        temps_cpu.append(delta)
        taille.append(t)
            
        verif = (np.dot(maa, Question2(maa, mbb)) - mbb)
        veriferreur = abs(np.sum(verif))
        print(veriferreur)
           
        erreur.append(veriferreur)
            
        t += 25
        print("DONE")
    tf = time.time()
    z = tf - td
    print(z)
    print("END")
        
    plt.plot(np.array(taille), np.array(temps_cpu), label = "Temps CPU")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Temps (s)")
    plt.legend()
    plt.title("Evolution du temps de calcul en fonction de la dimension de la matrice")
    plt.show()
    
    plt.plot(np.array(taille), np.array(erreur), label = "Erreur")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Erreur")
    plt.title("Evolution de l'erreur en fonction de la dimension de la matrice")
    plt.legend()
    
    plt.show()

    

ma = np.array([[3, 2, -1, 4], [-3, -4, 4, -2], [6, 2, 2, 7], [9, 4, 2, 18]])
mb = np.array([[4], [-5], [-2], [13]])

maa = np.array([[2, 5, 6], [4, 11, 9], [-2, -8, 7]])
mbb = np.array([[7], [12], [3]])

ta = np.array([[1, 1, 1, 1], [2, 4, -3, 2], [-1, -1, 0, -3], [1, -1, 4, 9]])
tb = np.array([[1], [1], [2], [-8]])


#print(TestQuestion2())


#############Partie3################

def PivotPartiel(A, B):
    Aaug = np.concatenate((A, B), axis = 1)
    n, m = np.shape(Aaug)
    x = np.zeros(n)
    ReductionGauss(Aaug)    
    for i in range(n - 1, -1, -1):
        somme = 0
        for j in range(i, n):
            for k in range(0, n - 1):
                if abs(Aaug[k, j]) < abs(Aaug[i, j]):
                    memoire = Aaug[k, j]
                    Aaug[k, j] = Aaug[i, j]
                    Aaug[i, j] = memoire
                print(Aaug)
                somme = somme + Aaug[i, j] * x[j]
                x[i] = (Aaug[i, n] - somme / Aaug[i, i])
    return x




#############Analyse##################
    
def TestMethode():
    t = 1
    temps_gauss = []
    temps_lu = []
    temps_solve = []
    erreur_gauss = []
    erreur_lu = []
    erreur_solve = []
    taille = []
    x = time.time()
    while t <= 2501:
        
        mata = fonction_matrice_A(t)
        matb = fonction_matrice_B(t)
        
        taille.append(t)
        
        debut_gauss = time.process_time()
        XG = Gauss(mata, matb)
        fin_gauss = time.process_time()
        delta_gauss = fin_gauss - debut_gauss
        temps_gauss.append(delta_gauss)
        
        axg = np.dot(mata, XG)
        test_gauss = axg - matb
        
        n, m = np.shape(test_gauss)
        diag = 0
        for i in range(n):
            diag += abs(test_gauss[i, i])
        erreur_gauss.append(diag)
        
        
        
        debut_lu = time.process_time()
        Question2(mata, matb)
        fin_lu = time.process_time()
        delta_lu = fin_lu - debut_lu
        temps_lu.append(delta_lu)
        
        verif = (np.dot(mata, Question2(mata, matb)) - matb)
        veriferreur = abs(np.sum(verif))
        erreur_lu.append(veriferreur)
        
        debut_solve = time.process_time()
        xs = np.linalg.solve(mata, matb)
        fin_solve = time.process_time()
        delta_solve = fin_solve - debut_solve
        print("DONE")
        temps_solve.append(delta_solve)
        
        axs = np.dot(mata, xs)
        test_solve = axs - matb
        somme_solve = abs(np.sum(test_solve))
        erreur_solve.append(somme_solve)
        
        t += 50
        
    plt.plot(np.array(taille), np.array(temps_gauss), label = "Gauss")
    plt.plot(np.array(taille), np.array(temps_lu), label = "LU")
    plt.plot(np.array(taille), np.array(temps_solve), label = "Solve")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Temps (s)")
    plt.legend()
    plt.title("Evolution du temps de calcul en fonction de la dimension de la matrice")
    plt.show()
    
    plt.plot(np.array(taille), np.array(erreur_gauss), label = "Gauss")
    plt.plot(np.array(taille), np.array(erreur_lu), label = "LU")
    plt.plot(np.array(taille), np.array(erreur_solve), label = "Solve")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Erreur")
    plt.title("Evolution de l'erreur en fonction de la dimension de la matrice")
    plt.legend()
    
    plt.show()
    y = time.time()
    z = y-x
    print(z)
        
    
def linalg():
    t = 25
    taille = []
    erreur_solve= []
    temps_solve = []
    x = time.time()
    while t <= 2500:
    
        mata = fonction_matrice_A(t)
        matb = fonction_matrice_B(t)
        
        debut_solve = time.process_time()
        xs = np.linalg.solve(mata, matb)
        fin_solve = time.process_time()
        delta_solve = fin_solve - debut_solve
        print("Taille:", t, "Temps:", delta_solve)
        print("DONE")
        temps_solve.append(delta_solve)
        
        """axs = np.dot(mata, xs)
        test_solve = axs - matb
        somme_solve = abs(np.sum(test_solve))
        erreur_solve.append(somme_solve)"""
        
        taille.append(t)
        
        t += 25
        
        
        
    y = time.time()
    z = y - x
    print(z)
    plt.plot(np.array(taille), np.array(temps_solve), label = "Temps")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Temps (s)")
    plt.legend()
    plt.title("Evolution du temps de calcul en fonction de la dimension de la matrice")
    plt.show()
    

    plt.plot(np.array(taille), np.array(erreur_solve), label = "Erreur")
    plt.xlabel("Dimension de la matrice (n)")
    plt.ylabel("Erreur")
    plt.title("Evolution de l'erreur en fonction de la dimension de la matrice")
    plt.legend()
    plt.show
    
linalg()
    
        
        
    
    



import os
import numpy as np
import sys
from enum import Enum
import time
from collections import Counter
import time

#VIZ
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


# effectue viz avec grille
def viz(matrix):

    # Create discrete colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'black'])
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(matrix, cmap=cmap, norm=norm)

    # Draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    # Set integer ticks for x and y axes
    size = matrix.shape[0]
    ax.set_xticks(np.arange(0, size, 1))
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_xticklabels(np.arange(1, size + 1))  # Increment the labels by 1
    ax.set_yticklabels(np.arange(1, size + 1))  # Increment the labels by 1

    plt.show()

print("\n", "CONSOLE", "\n---------------")

# Booleans
show_print = False
show_values = True

run_pers_file = False
# Viz with grid
show_viz = True
# Show columns and rows sequences
show_seq = False
show_progress_matrix = True

# Obtenir elements du repertoire
cur_dir = os.getcwd()
instances_dir = os.path.join(cur_dir, "instances")
instances_pers_dir = os.path.join(cur_dir, "instances_pers")
els = os.listdir(instances_dir)

if run_pers_file:
    test_file = os.path.join(instances_pers_dir,"pers_0.txt")
else:
    test_file = os.path.join(instances_dir,"15.txt")

# obtenir dimensions de la grille
def get_seq_grid(test_file):
    print(f"Fichier en lecture: {test_file}\n")
    N = 0
    M = 0
    reading_rows = True

    row_seq_list = []
    col_seq_list = []

    with open(test_file, "r") as file:
        for line in file:
            line = line.strip()
            #print(line, end="")
            if "#" in line:
                reading_rows = False
            else:
                seq_list = [int(x) for x in line.split()]
                if reading_rows:
                    N = N+1
                    row_seq_list.append(seq_list)
                else:
                    M = M+1
                    col_seq_list.append(seq_list)
    return (N, M), row_seq_list, col_seq_list

(N, M), row_seq_list, col_seq_list = get_seq_grid(test_file)

if show_values:
    print("M =", M, "/", "N =", N)
    print("\n")

def T(j,l, row_seq, dict, filled_cases_list, show_print):
    # Cas 1
    if (j,l) in dict:
         return dict[(j,l)]

    if l== 0:
        if show_print: print(f"Cas 1 : (j,l) = {j,l}")
        # verifier qu'il n'y a aucune case noire
        for k in range(j+1):
            if filled_cases_list[k] == 1:
                dict[(j,l)] = False
                return False
        dict[(j,l)] = True
        return True
    
    # on recupere dernier element de sequence
    s_l = row_seq[l-1]
    if show_print: print(f"T({j}, {l}), s_l = {s_l}")
    
    if j < s_l - 1:
        if show_print: print("Case 2-a")
        dict[(j,l)] = False
        return False
    elif j == s_l - 1:
        if show_print: print("Case 2-b")
        if l == 1:
            # il faut verifier qu'il n'y a aucune case deja coloriee 
            for k in range(j+1):
                # si une case est blanche, pas possible
                if filled_cases_list[k] == 0:
                    dict[(j,l)] = False
                    return False
            dict[(j,l)] = True
            return True
        elif l > 0:
            dict[(j,l)] = False
            return False
        else:
            print("Error l <= 0")
    elif j > s_l - 1:
        if show_print: print("Case 2-c")
        # Utilisation de memoisation
        
        if show_print: print("j=",j)
        # si case est blanche
        if filled_cases_list[j] == 0:
            if show_print: print("cas 2-c-e")
            result = T(j-1, l, row_seq, dict, filled_cases_list, show_print)
        # si case est noire
        elif filled_cases_list[j] == 1:
            if show_print: print("cas 2-c-f")
            # verifier qu'il y a case blanche
            valid_case = True
            for i in range(j-s_l+1, j+1):
                if filled_cases_list[i] == 0:
                    valid_case = False
                    if show_print: print("Cas: 2-c-a")
            if filled_cases_list[j-s_l] == 1:
                valid_case = False
                if show_print: print("Cas 2-c-b")
            if valid_case:
                result = T(j - s_l - 1, l - 1, row_seq, dict, filled_cases_list, show_print)
            else:
                result = False
        # il faut traiter les deux cas
        elif filled_cases_list[j] == -1:
            # verifier qu'il y a case blanche
            valid_case = True
            for i in range(j-s_l+1, j+1):
                if filled_cases_list[i] == 0:
                    valid_case = False
                    if show_print: print("Cas: 2-c-a")
            if filled_cases_list[j-s_l] == 1:
                valid_case = False
                if show_print: print("Cas 2-c-b")
            if valid_case:
                result = T(j-1, l, row_seq, dict, filled_cases_list, show_print) or T(j - s_l - 1, l - 1, row_seq, dict, filled_cases_list, show_print)
            else:
                result = T(j-1, l, row_seq, dict, filled_cases_list, show_print)

        dict[(j,l)] = result
        return result
    else:
        print("erreur dernier cas")

# test de la ligne
def test_row(i, row_seq_list, filled_row_list, M, show_print):
    row_seq = row_seq_list[i]
    k = len(row_seq)
    # test ligne entiere avec toute la sequence
    row_values_dict = dict()
    is_valid = T(M-1, k, row_seq, row_values_dict, filled_row_list, show_print) #M-1
    return is_valid

# test de la colonne
def test_col(i, col_seq_list, filled_col_list, N, show_print) :
    col_seq = col_seq_list[i]
    k = len(col_seq)
    # test ligne entiere avec toute la sequence
    col_values_dict = dict()
    is_valid = T(N-1, k, col_seq, col_values_dict, filled_col_list, show_print) #N-1
    return is_valid

# Complexite: O(NM)
def is_grid_colored(matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            el = matrix[row][col]
            if el == -1:
                return False
    return True

def coloration(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print):
    lignes_a_voir = set([i for i in range(N)])
    cols_a_voir = set([i for i in range(M)])
    progress_matrix = filled_cases_grid.copy()

    while len(lignes_a_voir) > 0 or len(cols_a_voir) > 0:
        if show_print: print("\n")

        for i in lignes_a_voir:
        
            for j in range(M):
                if show_print: print("LIGNE",i, "COL",j)
                if show_print: print('---------')
                if show_print: print(progress_matrix)
                if show_print: print("\n")
                if progress_matrix[i,j] == -1:
                    # updates par couleur
                    white_filled_row = progress_matrix[i, :].copy()
                    black_filled_row = progress_matrix[i, :].copy()

                    white_filled_row[j] = 0
                    black_filled_row[j] = 1

                    if show_print: print(f"white_filled_row: {white_filled_row}\nblack_filled_row: {black_filled_row}")
                    if show_print: print(row_seq_list[i])
                    test_white = test_row(i, row_seq_list, white_filled_row, M, show_print)
                    test_black = test_row(i, row_seq_list, black_filled_row, M, show_print)
                    if show_print: print(f"test_white: {test_white} | test_black: {test_black}")

                    # detection impossibilite: puzzle n'a pas de solution
                    if test_white == False and test_black == False:
                        return (False, None)
                    # test positif donc on modifie la matrice
                    if test_white == True and test_black == False:
                        progress_matrix[i][j] = 0
                        cols_a_voir = cols_a_voir.union(set([j]))
                    if test_white == False and test_black == True:
                        progress_matrix[i][j] = 1
                        cols_a_voir = cols_a_voir.union(set([j]))

                else:
                    rowt = progress_matrix[i, :].copy()
                    if not test_row(i, row_seq_list, rowt, M, show_print):
                        return (False, None)

                # si test_white == True and test_black == True: on ne peut rien deduire donc reste -1
                
        if show_print: print("cols a voir:", cols_a_voir)
        lignes_a_voir = set()

        if show_print: print('PARTIE AVEC COLONNES ---- ')

        for j in cols_a_voir:
            
            for i in range(N):
                if show_print: print("COL",j, "LIGNE", i)
                if show_print: print(progress_matrix)
                if show_print: print('\n')
                # updates par couleur
                if progress_matrix[i,j] == -1:
                    white_filled_row = progress_matrix[:, j].copy()
                    black_filled_row = progress_matrix[:, j].copy()
            
                    white_filled_row[i] = 0
                    black_filled_row[i] = 1
                    
                    if show_print: print(f"white_filled_row: {white_filled_row}\nblack_filled_row: {black_filled_row}")
                    if show_print: print(col_seq_list[i])
                    test_white = test_col(j, col_seq_list, white_filled_row, N, show_print)
    
                    test_black = test_col(j, col_seq_list, black_filled_row, N, show_print)
                    if show_print: print(f"test_white: {test_white} | test_black: {test_black}")

                    # detection impossibilite: puzzle n'a pas de solution
                    if test_white == False and test_black == False:
                        return (False, None)
                    # test positif donc on modifie la matrice
                    if test_white == True and test_black == False:
                        progress_matrix[i][j] = 0
                        lignes_a_voir = lignes_a_voir.union(set([i]))
                    if test_white == False and test_black == True:
                        progress_matrix[i][j] = 1
                        lignes_a_voir = lignes_a_voir.union(set([i]))

                else:
                   colt = progress_matrix[:, j].copy()
                   if not test_col(j, col_seq_list, colt, N, show_print):
                            return (False, None)

                 # si test_white == True and test_black == True: on ne peut rien deduire donc reste -1
        cols_a_voir = set()
        if show_print: print("lignes a voir:", lignes_a_voir)
        

    # il faut vérifier si toutes les cases sont coloriées
    if is_grid_colored(progress_matrix):
        return (True, progress_matrix)
    else: # correspond au cas : "ne sait pas"
        return (-1, progress_matrix)













# RESOLUTION COMPLETE

# Partiellement coloriee en entree
def colorier_et_propager(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print, i, j, c):
    progress_matrix = filled_cases_grid.copy()
    progress_matrix[i][j] = c
        
    #lignes a voir: {i}, colonnes a voir:{j}
    lignes_a_voir = set([i])
    cols_a_voir = set([j])


    #idem coloration
    while len(lignes_a_voir) > 0 or len(cols_a_voir) > 0:
        if show_print: print("\n")

        for i in lignes_a_voir:
        
            for j in range(M):
                if show_print: print("LIGNE",i, "COL",j)
                if show_print: print('---------')
                if show_print: print(progress_matrix)
                if show_print: print("\n")
                if progress_matrix[i,j] == -1:
                    # updates par couleur
                    white_filled_row = progress_matrix[i, :].copy()
                    black_filled_row = progress_matrix[i, :].copy()

                    white_filled_row[j] = 0
                    black_filled_row[j] = 1

                    if show_print: print(f"white_filled_row: {white_filled_row}\nblack_filled_row: {black_filled_row}")
                    if show_print: print(row_seq_list[i])
                    test_white = test_row(i, row_seq_list, white_filled_row, M, show_print)
                    test_black = test_row(i, row_seq_list, black_filled_row, M, show_print)
                    if show_print: print(f"test_white: {test_white} | test_black: {test_black}")

                    # detection impossibilite: puzzle n'a pas de solution
                    if test_white == False and test_black == False:
                        return (False, None)
                    # test positif donc on modifie la matrice
                    if test_white == True and test_black == False:
                        progress_matrix[i][j] = 0
                        cols_a_voir = cols_a_voir.union(set([j]))
                    if test_white == False and test_black == True:
                        progress_matrix[i][j] = 1
                        cols_a_voir = cols_a_voir.union(set([j]))
                else:
                    rowt = progress_matrix[i, :].copy()
                    if not test_row(i, row_seq_list, rowt, M, show_print):
                        return (False, None)

                # si test_white == True and test_black == True: on ne peut rien deduire donc reste -1
                
        if show_print: print("cols a voir:", cols_a_voir)
        lignes_a_voir = set()

        if show_print: print('PARTIE AVEC COLONNES ---- ')

        for j in cols_a_voir:
            
            for i in range(N):
                if show_print: print("COL",j, "LIGNE", i)
                if show_print: print(progress_matrix)
                if show_print: print('\n')
                # updates par couleur
                if progress_matrix[i,j] == -1:
                    white_filled_row = progress_matrix[:, j].copy()
                    black_filled_row = progress_matrix[:, j].copy()
            
                    white_filled_row[i] = 0
                    black_filled_row[i] = 1
                    
                    if show_print: print(f"white_filled_row: {white_filled_row}\nblack_filled_row: {black_filled_row}")
                    if show_print: print(col_seq_list[i])
                    test_white = test_col(j, col_seq_list, white_filled_row, N, show_print)
    
                    test_black = test_col(j, col_seq_list, black_filled_row, N, show_print)
                    if show_print: print(f"test_white: {test_white} | test_black: {test_black}")

                    # detection impossibilite: puzzle n'a pas de solution
                    if test_white == False and test_black == False:
                        return (False, None)
                    # test positif donc on modifie la matrice
                    if test_white == True and test_black == False:
                        progress_matrix[i][j] = 0
                        lignes_a_voir = lignes_a_voir.union(set([i]))
                    if test_white == False and test_black == True:
                        progress_matrix[i][j] = 1
                        lignes_a_voir = lignes_a_voir.union(set([i]))
                else:
                    colt = progress_matrix[:, j].copy()
                    if not test_col(j, col_seq_list, colt, N, show_print):
                        return (False, None)


                 # si test_white == True and test_black == True: on ne peut rien deduire donc reste -1
        cols_a_voir = set()
        if show_print: print("lignes a voir:", lignes_a_voir)
        

    # il faut vérifier si toutes les cases sont coloriées
    if is_grid_colored(progress_matrix):
        return (True, progress_matrix)
    else: # correspond au cas : "ne sait pas"
        if show_print: print("CAS 13")
        return (-1, progress_matrix)
    

def enumeration(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print):
    bool, progress_matrix = coloration(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print)

    # on sait que ce n'est pas possible
    if bool == False:
        return (False, None)
    # cas d'indetermination

    resbool1, resmat1 = enum_rec(N, M, row_seq_list, col_seq_list, progress_matrix, show_print, 0, 0)
    if resbool1 == True:
        return (True, resmat1)
    resbool2, resmat2 = enum_rec(N, M, row_seq_list, col_seq_list, progress_matrix, show_print, 0, 1)
    if resbool2 == True:
        return (resbool2, resmat2)
    return (False, None)

# k: indice de case
# c: couleur
def enum_rec(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print, k, c):
    # toutes les cases sont coloriees: on a termine
    if k == N*M:
        return (True, filled_cases_grid)
    
    i = k // M
    j = k % M
    bool, progress_matrix = colorier_et_propager(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print, i, j, c)

    if bool == False:
        return (False, None)
    elif bool == True:
        return (True, progress_matrix)
   
    # bool == -1

    #recherche de prochaine case indéterminée à partir de k+1
    t = k + 1
    while t < N*M:
        i = t // M
        j = t % M
        if progress_matrix[i][j] == -1: # case indeterminee
            break
        t = t + 1
    resbool1, resmat1 = enum_rec(N, M, row_seq_list, col_seq_list, progress_matrix, show_print, t, 0)
    if resbool1 == True:
        return (True, resmat1)
    resbool2, resmat2 = enum_rec(N, M, row_seq_list, col_seq_list, progress_matrix, show_print, t, 1)
    if resbool2 == True:
        return (resbool2, resmat2)
    return (False, None)
    

# representation avec les lignes
filled_cases_grid = np.full((N,M), -1)

#variable complete est égale à True si on veut appliquer algorithme de résolution complète, sinon on applique résolution partielle
complete = True
start_time = time.time()
if complete == True:
    bool, progress_matrix = enumeration(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print=show_print)
else:
    bool, progress_matrix = coloration(N, M, row_seq_list, col_seq_list, filled_cases_grid, show_print=show_print)

print("is_Colored:", is_grid_colored(progress_matrix))
end_time = time.time()
print(f"Duration: {end_time - start_time: .4f}")

# afficher les séquences
if show_seq:
    print(col_seq_list)
    print(row_seq_list)

if bool == -1:
    print("Cas NeSaitPas")
elif bool == False:
    print("Cas Impossibilité")
elif bool == True:
    if show_progress_matrix:
        print(progress_matrix)
    if show_viz:
        viz(progress_matrix)


import streamlit as st
import ezc3d
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Score eGVI", layout="centered")
st.title("🦿 Score eGVI - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)

if uploaded_files:
    selected_file_dynamique1 = st.selectbox("Choisissez un fichier dynamique 1 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique2 = st.selectbox("Choisissez un fichier dynamique 2 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique3 = st.selectbox("Choisissez un fichier dynamique 3 pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique1.read())
        tmp1_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique2.read())
        tmp2_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique3.read())
        tmp3_path = tmp.name
        
    # Valeur population contrôle Arnaud Gouelle
    cn = [0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90]
    cn = np.array(cn)
    m_ln_CTRL = 1.386573
    sd_ln_CTRL = 0.619334
    Projection_CTRL = 20.38

if st.button("Lancer le calcul du score eGVI"):
    try:
        # ==============================================================================
        # CONFIGURATION : LISTE DES FICHIERS A TRAITER
        # ==============================================================================
        # Remplacez les chemins ci-dessous par vos 3 fichiers c3d
        liste_fichiers = [
            tmp1_path,
            tmp2_path,
            tmp3_path
        ]
        
        # Dictionnaires pour stocker les "Différences" de tous les fichiers combinés
        # C'est ce qui servira au calcul final de l'EGVI
        global_diffs_left = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        global_diffs_right = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        
        # ==============================================================================
        # FONCTIONS UTILITAIRES (Pour alléger la boucle principale)
        # ==============================================================================
        
        def find_toe_offs_from_toes(z_toe_data, cycle_tuples, threshold_clearance=20.0):
            """ Détecte le Toe Off en cherchant quand l'orteil remonte après avoir été au sol. """
            detected_tos = []
            for (start_frame, end_frame) in cycle_tuples:
                z_segment = z_toe_data[start_frame:end_frame]
                if len(z_segment) == 0: continue
                idx_min = np.argmin(z_segment)
                min_val = z_segment[idx_min]
                post_min_segment = z_segment[idx_min:]
                candidates = np.where(post_min_segment > (min_val + threshold_clearance))[0]
                if len(candidates) > 0:
                    to_frame = start_frame + idx_min + candidates[0]
                    detected_tos.append(to_frame)
            return detected_tos
        
        def calculate_diffs(raw_values, normalization_mean=None):
            """
            Calcule les différences absolues entre pas consécutifs normalisés.
            Respecte la logique de votre script : ((valeur/facteur)/moyenne*100)
            """
            if len(raw_values) < 2:
                return []
        
            arr = np.array(raw_values)
        
            # Si la moyenne n'est pas fournie, on utilise la moyenne locale
            if normalization_mean is None:
                normalization_mean = np.mean(arr)
        
            normalized_p = []
            diffs = []
        
            # Note: Votre script divise StepLength par 10, mais pas Time/Stance/Single.
            # Pour généraliser, on assume que raw_values est déjà dans la bonne unité
            # ou que la division par 10 se fait AVANT l'appel pour StepLength.
            # Cependant, pour respecter STRICTEMENT votre boucle :
            # StepLenPleft.append((i/10)/np.mean(step_lengths_left)*100)
            # Je vais traiter la division par 10 à l'extérieur pour StepLength.
        
            for val in arr:
                normalized_p.append((val) / normalization_mean * 100)
        
            for i in range(len(normalized_p) - 1):
                diff = abs(normalized_p[i+1] - normalized_p[i])
                diffs.append(diff)
        
            return diffs
        
        # ==============================================================================
        # BOUCLE PRINCIPALE SUR LES FICHIERS
        # ==============================================================================
        
        print(f"Début du traitement de {len(liste_fichiers)} fichiers...")
        
        for fichier in liste_fichiers:
            if not os.path.exists(fichier):
                print(f"ATTENTION : Fichier introuvable {fichier}, passage au suivant.")
                continue
        
            print(f"\nTraitement du fichier : {fichier}")
        
            # 1. Chargement
            acq1 = ezc3d.c3d(fichier)
            labels = acq1['parameters']['POINT']['LABELS']['value']
            freq = acq1['header']['points']['frame_rate']
            points = acq1['data']['points']
        
            # Paramètres de base
            axis_ap = 0 # Axe de progression (X)
        
            # 2. Détection Cycles (LHEE / RHEE)
            lhee_valid_cycles = []
            rhee_valid_cycles = []
            lhee_cycle_start_indices = []
            rhee_cycle_start_indices = []
        
            # --- GAUCHE ---
            if "LHEE" in labels:
                idx_lhee = labels.index("LHEE")
                z_lhee = points[2, idx_lhee, :]
                inverted_z = -z_lhee
                min_distance = int(freq * 0.8)
                peaks, _ = find_peaks(inverted_z, distance=min_distance, prominence=1)
                lhee_cycle_start_indices = peaks[:-1]
                lhee_cycle_end_indices = peaks[1:]
                min_lhee_cycle_duration = int(0.5 * freq)
                lhee_valid_cycles = [
                    (start, end) for start, end in zip(lhee_cycle_start_indices, lhee_cycle_end_indices)
                    if (end - start) >= min_lhee_cycle_duration
                ]
        
            # --- DROITE ---
            if "RHEE" in labels:
                idx_rhee = labels.index("RHEE")
                z_rhee = points[2, idx_rhee, :]
                inverted_z = -z_rhee
                min_distance = int(freq * 0.8)
                peaks, _ = find_peaks(inverted_z, distance=min_distance, prominence=1)
                rhee_cycle_start_indices = peaks[:-1]
                rhee_cycle_end_indices = peaks[1:]
                min_rhee_cycle_duration = int(0.5 * freq)
                rhee_valid_cycles = [
                    (start, end) for start, end in zip(rhee_cycle_start_indices, rhee_cycle_end_indices)
                    if (end - start) >= min_rhee_cycle_duration
                ]
        
            # 3. Calcul Step Length (Séparé)
            step_lengths_left = []
            step_lengths_right = []
        
            all_events = []
            for frame in lhee_cycle_start_indices: all_events.append((frame, 'Left'))
            for frame in rhee_cycle_start_indices: all_events.append((frame, 'Right'))
            all_events.sort(key=lambda x: x[0])
        
            if "LHEE" in labels and "RHEE" in labels:
                idx_lhee = labels.index("LHEE")
                idx_rhee = labels.index("RHEE")
        
                for i in range(1, len(all_events)):
                    current_frame, current_side = all_events[i]
                    prev_frame, prev_side = all_events[i-1]
        
                    if current_side != prev_side:
                        pos_lhee = points[axis_ap, idx_lhee, current_frame]
                        pos_rhee = points[axis_ap, idx_rhee, current_frame]
                        step_len = abs(pos_lhee - pos_rhee)/10 # Division par 10 conservée
        
                        if current_side == 'Left': step_lengths_left.append(step_len)
                        else: step_lengths_right.append(step_len)
        
            # 4. Calcul Step Time
            step_times_left = []
            step_times_right = []
        
            # On réutilise all_events
            for i in range(1, len(all_events)):
                current_frame, current_side = all_events[i]
                prev_frame, prev_side = all_events[i-1]
        
                if current_side != prev_side:
                    step_time_seconds = (current_frame - prev_frame) / freq
                    if current_side == 'Left': step_times_left.append(step_time_seconds)
                    else: step_times_right.append(step_time_seconds)
        
            # 5. Détection Toe Offs (LTOE / RTOE)
            lhee_toe_offs = []
            rhee_toe_offs = []
        
            if "LTOE" in labels and len(lhee_valid_cycles) > 0:
                idx_ltoe = labels.index("LTOE")
                z_ltoe = points[2, idx_ltoe, :]
                lhee_toe_offs = find_toe_offs_from_toes(z_ltoe, lhee_valid_cycles)
        
            if "RTOE" in labels and len(rhee_valid_cycles) > 0:
                idx_rtoe = labels.index("RTOE")
                z_rtoe = points[2, idx_rtoe, :]
                rhee_toe_offs = find_toe_offs_from_toes(z_rtoe, rhee_valid_cycles)
        
            # 6. Single Support & Stance Time
            sst_left, sst_right = [], []
            stance_time_left, stance_time_right = [], []
        
            l_starts = [x[0] for x in lhee_valid_cycles]
            r_starts = [x[0] for x in rhee_valid_cycles]
        
            # SST & Stance GAUCHE
            for ic_frame in l_starts: # Stance starts at IC Left
                next_tos = [to for to in lhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        stance_time_left.append((to_frame - ic_frame) / freq)
        
            for to_frame in rhee_toe_offs: # SST Left starts at TO Right
                next_ics = [ic for ic in r_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        sst_left.append((next_ic - to_frame) / freq)
        
            # SST & Stance DROIT
            for ic_frame in r_starts: # Stance starts at IC Right
                next_tos = [to for to in rhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        stance_time_right.append((to_frame - ic_frame) / freq)
        
            for to_frame in lhee_toe_offs: # SST Right starts at TO Left
                next_ics = [ic for ic in l_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        sst_right.append((next_ic - to_frame) / freq)
        
            # 7. Velocity (via Stride)
            velocity_left = []
            velocity_right = []
        
            def get_velocity(cycles, marker_idx):
                vels = []
                for (start, end) in cycles:
                    dur = (end - start) / freq
                    p_start = points[axis_ap, marker_idx, start]
                    p_end = points[axis_ap, marker_idx, end]
                    dist = abs(p_end - p_start)/10
                    if dur > 0: vels.append(dist/dur)
                return vels
        
            if len(lhee_valid_cycles) > 0 and "LHEE" in labels:
                velocity_left = get_velocity(lhee_valid_cycles, idx_lhee)
            if len(rhee_valid_cycles) > 0 and "RHEE" in labels:
                velocity_right = get_velocity(rhee_valid_cycles, idx_rhee)
        
        
            # ==========================================================================
            # AGGRÉGATION DES DIFFÉRENCES (Le cœur de l'EGVI Multi-fichier)
            # ==========================================================================
            # On calcule les diffs POUR CE FICHIER et on les ajoute à la liste globale.
            # On normalise par la moyenne DE CE FICHIER (standard pour éviter les sauts d'offset entre fichiers)
            # Note : Pour StepLength, vos valeurs brutes sont déjà divisées par 10 dans 'step_lengths_left'
            # Mais dans votre script original, vous refaites (i/10) lors du calcul de diff.
            # ATTENTION : Dans mon bloc 3 ci-dessus, j'ai déjà divisé par 10 (step_len = ... / 10).
            # Donc ici, je passe les valeurs telles quelles, MAIS je dois gérer la division supplémentaire si votre logique l'exige.
            # Votre script : StepLenPleft.append((i/10)/mean*100). Or 'i' venait de step_lengths_left qui avait déjà /10 ?
            # Vérifions votre script :
            #   Bloc 4 : step_len = abs(...)/10. -> step_lengths_left contient des cm.
            #   Bloc Resultats : StepLenPleft.append((i/10)/mean*100). -> Vous redivisez par 10 ?
            #   Cela semble être une erreur potentielle dans le script original ou une conversion mm->cm->dm ?
            #   Je vais RESPECTER LA LOGIQUE ECRITE : Je vais diviser par 10 les valeurs entrantes pour StepLength.
        
            # --- GAUCHE ---
            # 1. Step Length (avec la re-division par 10 spécifique à votre script)
            sl_l_processed = [x/10 for x in step_lengths_left]
            global_diffs_left['StepLen'].extend(calculate_diffs(sl_l_processed))
        
            # 2. Step Time
            global_diffs_left['StepTime'].extend(calculate_diffs(step_times_left))
        
            # 3. Stance Time
            global_diffs_left['StanceTime'].extend(calculate_diffs(stance_time_left))
        
            # 4. Single Support
            global_diffs_left['SingleSup'].extend(calculate_diffs(sst_left))
        
            # 5. Velocity
            global_diffs_left['Velocity'].extend(calculate_diffs(velocity_left))
        
        
            # --- DROITE ---
            # 1. Step Length
            sl_r_processed = [x/10 for x in step_lengths_right]
            global_diffs_right['StepLen'].extend(calculate_diffs(sl_r_processed))
        
            # 2. Step Time
            global_diffs_right['StepTime'].extend(calculate_diffs(step_times_right))
        
            # 3. Stance Time
            global_diffs_right['StanceTime'].extend(calculate_diffs(stance_time_right))
        
            # 4. Single Support
            global_diffs_right['SingleSup'].extend(calculate_diffs(sst_right))
        
            # 5. Velocity
            global_diffs_right['Velocity'].extend(calculate_diffs(velocity_right))
        
            print(f"  -> Cycles extraits (G/D) : {len(lhee_valid_cycles)} / {len(rhee_valid_cycles)}")
        
        print("\n" + "="*30)
        print(" CALCUL GLOBAL EGVI (Tous fichiers)")
        print("="*30)
        
        # ==============================================================================
        # CALCUL FINAL EGVI
        # ==============================================================================
        
        def calculer_egvi(donnees_sujet):
            # Coefficients (cn) - Arnaud Gouelle
            coeffs = np.array([0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90])
            mean_control_s_alpha = 20.37541132570365
            mean_ln_d_control = 1.3865728033714733
            sd_ln_d_control = 0.6193340454665202
        
            if len(donnees_sujet) != 10:
                return 0.0 # Erreur
        
            # Gestion des NaN si pas assez de données
            donnees_propres = [0.0 if np.isnan(x) else x for x in donnees_sujet]
        
            s_alpha_sujet = np.dot(coeffs, donnees_propres)
            diff = s_alpha_sujet - mean_control_s_alpha
            d = abs(diff) + 1
        
            if d < 0: ln_d = -(math.log(d))
            else: ln_d = math.log(d)
        
            z_score = (ln_d - mean_ln_d_control) / sd_ln_d_control
            egvi = 100 + (10 * z_score)
            return egvi
        
        # Fonction pour obtenir Moyenne et SD des listes de différences accumulées
        def get_stats_from_diffs(diff_list):
            arr = np.array(diff_list)
            if len(arr) == 0: return 0.0, 0.0
            return np.mean(arr), np.std(arr, ddof=1)
        
        # --- Construction du vecteur pour EGVI ---
        # Rappel de votre structure :
        # [ Mean_StepLenDiff*10, Mean_StepTimeDiff, Mean_StanceDiff, Mean_SingleDiff, Mean_VeloDiff,
        #   SD_StepLenDiff*10,   SD_StepTimeDiff,   SD_StanceDiff,   SD_SingleDiff,   SD_VeloDiff ]
        
        # DROITE
        m_sl_r, sd_sl_r = get_stats_from_diffs(global_diffs_right['StepLen'])
        m_st_r, sd_st_r = get_stats_from_diffs(global_diffs_right['StepTime'])
        m_sta_r, sd_sta_r = get_stats_from_diffs(global_diffs_right['StanceTime'])
        m_ss_r, sd_ss_r = get_stats_from_diffs(global_diffs_right['SingleSup'])
        m_vel_r, sd_vel_r = get_stats_from_diffs(global_diffs_right['Velocity'])
        
        valeurs_sujet_D = [
            m_sl_r,  m_st_r,  m_sta_r,  m_ss_r,  m_vel_r,
            sd_sl_r, sd_st_r, sd_sta_r, sd_ss_r, sd_vel_r
        ]
        
        # GAUCHE
        m_sl_l, sd_sl_l = get_stats_from_diffs(global_diffs_left['StepLen'])
        m_st_l, sd_st_l = get_stats_from_diffs(global_diffs_left['StepTime'])
        m_sta_l, sd_sta_l = get_stats_from_diffs(global_diffs_left['StanceTime'])
        m_ss_l, sd_ss_l = get_stats_from_diffs(global_diffs_left['SingleSup'])
        m_vel_l, sd_vel_l = get_stats_from_diffs(global_diffs_left['Velocity'])
        
        valeurs_sujet_G = [
            m_sl_l,  m_st_l,  m_sta_l,  m_ss_l,  m_vel_l,
            sd_sl_l, sd_st_l, sd_sta_l, sd_ss_l, sd_vel_l
        ]
        
        # Calcul Final
        egvi_resultat_D = calculer_egvi(valeurs_sujet_D)
        egvi_resultat_G = calculer_egvi(valeurs_sujet_G)
        EGVItot = (egvi_resultat_D + egvi_resultat_G)/2
        
        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"Nombre total de cycles (Approx) : {len(global_diffs_left['StepLen'])} + {len(global_diffs_right['StepLen'])}")
        st.write(f"**Score eGVI** : {EGVItot:.2f}")
        st.write(f"**Score eGVI droit** : {egvi_resultat_D:.2f}")
        st.write(f"**Score eGVI gauche** : {egvi_resultat_G:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 98 et 102. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
       
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")

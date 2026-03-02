import streamlit as st
import ezc3d
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, hilbert, find_peaks
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

st.set_page_config(page_title="Score eGVI", layout="centered")
st.title("🦿 Score eGVI - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d d'essai dynamique et un fichier statique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique1 = st.selectbox("Choisissez un fichier dynamique 1 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique2 = st.selectbox("Choisissez un fichier dynamique 2 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique3 = st.selectbox("Choisissez un fichier dynamique 3 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique4 = st.selectbox("Choisissez un fichier dynamique 4 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique5 = st.selectbox("Choisissez un fichier dynamique 5 pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique1.read())
        tmp1_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique2.read())
        tmp2_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique3.read())
        tmp3_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique4.read())
        tmp4_path = tmp.name
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique5.read())
        tmp5_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp6_path = tmp.name
        
    # Valeur population contrôle Arnaud Gouelle
    cn = [0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90]
    cn = np.array(cn)
    m_ln_CTRL = 1.386573
    sd_ln_CTRL = 0.619334
    Projection_CTRL = 20.38

if st.button("Lancer le calcul du score eGVI"):
    try:
        acq_stat = ezc3d.c3d(tmp6_path)
        pts_stat = acq_stat['data']['points']
        lbl_stat = acq_stat['parameters']['POINT']['LABELS']['value']
    
        iLPSI, iRPSI = lbl_stat.index('LPSI'), lbl_stat.index('RPSI')
        iLANK, iRANK = lbl_stat.index('LANK'), lbl_stat.index('RANK')
        iLASI, iRASI = lbl_stat.index('LASI'), lbl_stat.index('RASI')
    
        LgJambeR = np.mean(np.linalg.norm(pts_stat[:, iRANK, :] - pts_stat[:, iRPSI, :], axis=0))
        LgJambeL = np.mean(np.linalg.norm(pts_stat[:, iLANK, :] - pts_stat[:, iLPSI, :], axis=0))
        LargeurBassin = np.mean(np.abs(pts_stat[1, iLASI, :] - pts_stat[1, iRASI, :]))
        LgJambe_Moy_m = ((LgJambeL + LgJambeR) / 2) / 1000
        # ==============================================================================
        # CONFIGURATION : LISTE DES FICHIERS A TRAITER
        # ==============================================================================
        # Remplacez les chemins ci-dessous par vos 3 fichiers c3d
        liste_fichiers = [
            tmp1_path,
            tmp2_path,
            tmp3_path,
            tmp4_path,
            tmp5_path
        ]
        
        # Dictionnaires pour stocker les "Différences" de tous les fichiers combinés
        global_diffs_left = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        global_diffs_right = {
            'StepLen': [], 'StepTime': [], 'StanceTime': [], 'SingleSup': [], 'Velocity': []
        }
        
        # ==============================================================================
        # FONCTIONS UTILITAIRES
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
            Étape 1 : Calcule les différences absolues entre pas consécutifs normalisés.
            """
            if len(raw_values) < 2:
                return []
        
            arr = np.array(raw_values)
        
            if normalization_mean is None:
                normalization_mean = np.mean(arr)
        
            normalized_p = []
            diffs = []
        
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
        
            axis_ap = 0 # Axe de progression (X)
        
            # 2. Détection Cycles (LHEE / RHEE)
            lhee_valid_cycles, rhee_valid_cycles = [], []
            lhee_cycle_start_indices, rhee_cycle_start_indices = [], []
        
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
        
            # 3. Calcul Step Length
            step_lengths_left, step_lengths_right = [], []
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
                        step_len = abs(pos_lhee - pos_rhee)
                        if current_side == 'Left': step_lengths_left.append(step_len/LgJambe_Moy_m)
                        else: step_lengths_right.append(step_len/LgJambe_Moy_m)
        
            # 4. Calcul Step Time
            step_times_left, step_times_right = [], []
            for i in range(1, len(all_events)):
                current_frame, current_side = all_events[i]
                prev_frame, prev_side = all_events[i-1]
                if current_side != prev_side:
                    step_time_seconds = (current_frame - prev_frame) / freq
                    if current_side == 'Left': step_times_left.append(step_time_seconds)
                    else: step_times_right.append(step_time_seconds)
        
            # 5. Détection Toe Offs
            lhee_toe_offs, rhee_toe_offs = [], []
            if "LTOE" in labels and len(lhee_valid_cycles) > 0:
                idx_ltoe = labels.index("LTOE")
                lhee_toe_offs = find_toe_offs_from_toes(points[2, idx_ltoe, :], lhee_valid_cycles)
            if "RTOE" in labels and len(rhee_valid_cycles) > 0:
                idx_rtoe = labels.index("RTOE")
                rhee_toe_offs = find_toe_offs_from_toes(points[2, idx_rtoe, :], rhee_valid_cycles)
        
            # 6. Single Support & Stance Time
            sst_left, sst_right = [], []
            stance_time_left, stance_time_right = [], []
        
            l_starts = [x[0] for x in lhee_valid_cycles]
            r_starts = [x[0] for x in rhee_valid_cycles]
        
            # SST & Stance GAUCHE
            for ic_frame in l_starts:
                next_tos = [to for to in lhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        stance_time_left.append((to_frame - ic_frame) / freq)
        
            for to_frame in rhee_toe_offs:
                next_ics = [ic for ic in r_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        sst_left.append((next_ic - to_frame) / freq)
        
            # SST & Stance DROIT
            for ic_frame in r_starts:
                next_tos = [to for to in rhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        stance_time_right.append((to_frame - ic_frame) / freq)
        
            for to_frame in lhee_toe_offs:
                next_ics = [ic for ic in l_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        sst_right.append((next_ic - to_frame) / freq)
        
            # 7. Velocity (via Stride)
            def get_velocity(cycles, marker_idx):
                vels = []
                for (start, end) in cycles:
                    dur = (end - start) / freq
                    dist = abs(points[axis_ap, marker_idx, end] - points[axis_ap, marker_idx, start]) / 10
                    if dur > 0: vels.append((dist/dur)/(sqrt(9.81*LgJambe_Moy_m)))
                return vels
        
            velocity_left, velocity_right = [], []
            if len(lhee_valid_cycles) > 0 and "LHEE" in labels:
                velocity_left = get_velocity(lhee_valid_cycles, idx_lhee)
            if len(rhee_valid_cycles) > 0 and "RHEE" in labels:
                velocity_right = get_velocity(rhee_valid_cycles, idx_rhee)
        
            # AGGRÉGATION DES DIFFÉRENCES
            global_diffs_left['StepLen'].extend(calculate_diffs(step_lengths_left))
            global_diffs_left['StepTime'].extend(calculate_diffs(step_times_left))
            global_diffs_left['StanceTime'].extend(calculate_diffs(stance_time_left))
            global_diffs_left['SingleSup'].extend(calculate_diffs(sst_left))
            global_diffs_left['Velocity'].extend(calculate_diffs(velocity_left))
        
            global_diffs_right['StepLen'].extend(calculate_diffs(step_lengths_right))
            global_diffs_right['StepTime'].extend(calculate_diffs(step_times_right))
            global_diffs_right['StanceTime'].extend(calculate_diffs(stance_time_right))
            global_diffs_right['SingleSup'].extend(calculate_diffs(sst_right))
            global_diffs_right['Velocity'].extend(calculate_diffs(velocity_right))
        
            print(f"  -> Cycles extraits (G/D) : {len(lhee_valid_cycles)} / {len(rhee_valid_cycles)}")
        
        print("\n" + "="*30)
        print(" CALCUL GLOBAL EGVI (Tous fichiers)")
        print("="*30)
        
        # ==============================================================================
        # CALCUL FINAL EGVI
        # ==============================================================================
        
        def calculer_egvi(donnees_sujet):
            """
            Applique la méthodologie eGVI sur les données de différences pas-à-pas.
            """
            # Étape 2 : Coefficients (poids) par paramètre
            coeffs = np.array([0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90])
            mean_control_s_alpha = 20.37541132570365
            mean_ln_d_control = 1.3865728033714733
            sd_ln_d_control = 0.6193340454665202
        
            if len(donnees_sujet) != 10:
                return 0.0, 0.0, 0.0
        
            donnees_propres = [0.0 if np.isnan(x) else x for x in donnees_sujet]
        
            # Étape 2 (Suite) : Pondération (Score brut)
            s_alpha_sujet = np.dot(coeffs, donnees_propres)
        
            # Différence par rapport à la population saine
            diff = s_alpha_sujet - mean_control_s_alpha
        
            # Étape 3 : Calcul de la distance absolue et de l'ajustement logarithmique ln(1+distance)
            distance_absolue = abs(diff)
            ln_d = math.log(1 + distance_absolue)
        
        
        
            # Étape 4 : Le Z-score
            z_score = (ln_d - mean_ln_d_control) / sd_ln_d_control
        
            # Étape 5 (Suite) : Score composite final, centré sur 100
            egvi = 100 + (10 * z_score)
        
            return egvi, ln_d, s_alpha_sujet
        
        def get_stats_from_diffs(diff_list):
            arr = np.array(diff_list)
            if len(arr) == 0: return 0.0, 0.0
            return np.mean(arr), np.std(arr, ddof=1)
        
        # --- Construction du vecteur pour EGVI ---
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
        egvi_resultat_D, ln_dd, s_alpha_sujetd = calculer_egvi(valeurs_sujet_D)
        egvi_resultat_G, ln_dg, s_alpha_sujetg = calculer_egvi(valeurs_sujet_G)
        EGVItot = (egvi_resultat_D + egvi_resultat_G) / 2
        
        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"\nRÉSULTATS EGVI AGREGÉS (sur {len(global_diffs_left['StepLen'])} G et {len(global_diffs_right['StepLen'])} D pas cumulés) :")
        st.write(f"Score EGVI Gauche : {egvi_resultat_G:.2f}")
        st.write(f"Score EGVI Droit  : {egvi_resultat_D:.2f}")
        st.write(f"Score EGVI Global : {EGVItot:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 95 et 105. Tout score en-dehors indique une atteinte à la variabilité de la marche.
        Une trop grande variabilité en cas de score supérieur à 105 et une manque de variabilitée en cas de score inférieur à 95.")
       
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")

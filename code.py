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
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier d'essai statique et un d'essai dynamique")
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
        # CONFIGURATION ET LISTE DES FICHIERS
        # ==============================================================================
        
        # Indiquez ici vos 3 fichiers C3D
        fichiers_c3d = [
            "/BureauR03.c3d",
            "/BureauR04.c3d",  # Exemple nom fichier 2
            "/BureauR05.c3d"   # Exemple nom fichier 3
        ]
        
        # Listes Globales pour accumuler les données de tous les fichiers
        # Step Length
        GLOBAL_sl_left = []
        GLOBAL_sl_right = []
        # Step Time
        GLOBAL_st_left = []
        GLOBAL_st_right = []
        # Stride Velocity
        GLOBAL_sv_left = []
        GLOBAL_sv_right = []
        # Supports
        GLOBAL_sst_left = []
        GLOBAL_sst_right = []
        GLOBAL_dst_left = []
        GLOBAL_dst_right = []
        # Stance
        GLOBAL_stance_left = []
        GLOBAL_stance_right = []
        
        # --- FONCTION DÉTECTION TOE OFF (Définie une fois au début) ---
        def find_toe_offs_from_toes(z_toe_data, cycle_tuples):
            """
            Détecte le Toe Off en cherchant quand l'orteil remonte après avoir été au sol.
            """
            detected_tos = []
            threshold_clearance = 20.0 
            
            for (start_frame, end_frame) in cycle_tuples:
                z_segment = z_toe_data[start_frame:end_frame]
                if len(z_segment) == 0: continue
        
                # 1. Trouver le min global dans le cycle
                idx_min = np.argmin(z_segment)
                min_val = z_segment[idx_min]
                
                # 2. Chercher quand ça remonte au-dessus du seuil APRES ce minimum
                post_min_segment = z_segment[idx_min:]
                candidates = np.where(post_min_segment > (min_val + threshold_clearance))[0]
                
                if len(candidates) > 0:
                    to_frame = start_frame + idx_min + candidates[0]
                    detected_tos.append(to_frame)
                    
            return detected_tos
        
        # --- FONCTION CALCUL STRIDE ---
        def process_strides(cycles, marker_idx, points_data, freq, axis):
            times = []
            lengths = []
            velocities = []
            
            for (start_frame, end_frame) in cycles:
                duration_frames = end_frame - start_frame
                cycle_time_s = duration_frames / freq
                
                pos_start = points_data[axis, marker_idx, start_frame]
                pos_end = points_data[axis, marker_idx, end_frame]
                cycle_len_mm = abs(pos_end - pos_start)
                
                if cycle_time_s > 0:
                    vel_mms = cycle_len_mm / cycle_time_s
                else:
                    vel_mms = 0
                    
                times.append(cycle_time_s)
                lengths.append(cycle_len_mm)
                velocities.append(vel_mms)
                
            return times, lengths, velocities
        
        # ==============================================================================
        # BOUCLE DE TRAITEMENT SUR LES 3 FICHIERS
        # ==============================================================================
        
        for fichier in fichiers_c3d:
            print(f"Traitement du fichier : {fichier} ...")
            
            if not os.path.exists(fichier):
                print(f"  ATTENTION: Fichier {fichier} introuvable. Passé.")
                continue
        
            # Chargement
            acq1 = ezc3d.c3d(fichier)
            labels = acq1['parameters']['POINT']['LABELS']['value']
            freq = acq1['header']['points']['frame_rate']
            points = acq1['data']['points']
            
            # ------------------------------------------------------------------
            # 1. DÉTECTION EVENTS (Heel Strikes)
            # ------------------------------------------------------------------
            
            # GAUCHE (LHEE)
            lhee_valid_cycles = []
            lhee_cycle_start_indices = []
            if "LHEE" in labels:
                idx_lhee = labels.index("LHEE")
                z_lhee = points[2, idx_lhee, :]
                inverted_z = -z_lhee
                min_distance = int(freq * 0.8)
                peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
                
                lhee_cycle_start_indices = peaks[:-1]
                lhee_cycle_end_indices = peaks[1:]
                min_lhee_cycle_duration = int(0.5 * freq)
                lhee_valid_cycles = [
                  (start, end) for start, end in zip(lhee_cycle_start_indices, lhee_cycle_end_indices)
                  if (end - start) >= min_lhee_cycle_duration
                ]
        
            # DROITE (RHEE)
            rhee_valid_cycles = []
            rhee_cycle_start_indices = []
            if "RHEE" in labels:
                idx_rhee = labels.index("RHEE")
                z_rhee = points[2, idx_rhee, :]
                inverted_z = -z_rhee
                min_distance = int(freq * 0.8)
                peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
                
                rhee_cycle_start_indices = peaks[:-1]
                rhee_cycle_end_indices = peaks[1:]
                min_rhee_cycle_duration = int(0.5 * freq)
                rhee_valid_cycles = [
                  (start, end) for start, end in zip(rhee_cycle_start_indices, rhee_cycle_end_indices)
                  if (end - start) >= min_rhee_cycle_duration
                ]
        
            # ------------------------------------------------------------------
            # 2. DÉTECTION TOE OFFS (LTOE / RTOE)
            # ------------------------------------------------------------------
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
        
            # ------------------------------------------------------------------
            # 3. CALCUL STEP LENGTH & STEP TIME (Logique alternée)
            # ------------------------------------------------------------------
            axis_ap = 1 
            all_events = []
        
            # Consolidation events pour ce fichier
            for frame in lhee_cycle_start_indices:
                all_events.append((frame, 'Left'))
            for frame in rhee_cycle_start_indices:
                all_events.append((frame, 'Right'))
            all_events.sort(key=lambda x: x[0])
        
            if "LHEE" in labels and "RHEE" in labels:
                idx_lhee = labels.index("LHEE")
                idx_rhee = labels.index("RHEE")
        
                for i in range(1, len(all_events)):
                    current_frame, current_side = all_events[i]
                    prev_frame, prev_side = all_events[i-1]
        
                    if current_side != prev_side:
                        # --- Step Length ---
                        pos_lhee = points[axis_ap, idx_lhee, current_frame]
                        pos_rhee = points[axis_ap, idx_rhee, current_frame]
                        step_len = abs(pos_lhee - pos_rhee)
        
                        # --- Step Time ---
                        delta_frames = current_frame - prev_frame
                        step_time_s = delta_frames / freq
        
                        # Stockage dans les listes globales
                        if current_side == 'Left':
                            GLOBAL_sl_left.append(step_len)
                            GLOBAL_st_left.append(step_time_s)
                        else:
                            GLOBAL_sl_right.append(step_len)
                            GLOBAL_st_right.append(step_time_s)
        
            # ------------------------------------------------------------------
            # 4. CALCUL STRIDE VELOCITY
            # ------------------------------------------------------------------
            if len(lhee_valid_cycles) > 0:
                _, _, v = process_strides(lhee_valid_cycles, idx_lhee, points, freq, axis_ap)
                GLOBAL_sv_left.extend(v)
        
            if len(rhee_valid_cycles) > 0:
                _, _, v = process_strides(rhee_valid_cycles, idx_rhee, points, freq, axis_ap)
                GLOBAL_sv_right.extend(v)
        
            # ------------------------------------------------------------------
            # 5. CALCUL TEMPS D'APPUI (Single/Double/Stance)
            # ------------------------------------------------------------------
            l_starts = [x[0] for x in lhee_valid_cycles]
            r_starts = [x[0] for x in rhee_valid_cycles]
        
            # --- Single Support & Double Support ---
            # SST GAUCHE & DST GAUCHE (Basé sur les cycles)
            for ic_frame in l_starts: # Cycles commençant par gauche
                # Double Support Gauche
                next_tos = [to for to in rhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (0.5 * freq):
                        GLOBAL_dst_left.append((to_frame - ic_frame) / freq)
            
            for to_frame in rhee_toe_offs:
                # Single Support Gauche (Swing droit)
                next_ics = [ic for ic in r_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        GLOBAL_sst_left.append((next_ic - to_frame) / freq)
        
            # SST DROIT & DST DROIT
            for ic_frame in r_starts:
                # Double Support Droit
                next_tos = [to for to in lhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (0.5 * freq):
                        GLOBAL_dst_right.append((to_frame - ic_frame) / freq)
        
            for to_frame in lhee_toe_offs:
                # Single Support Droit (Swing gauche)
                next_ics = [ic for ic in l_starts if ic > to_frame]
                if next_ics:
                    next_ic = next_ics[0]
                    if (next_ic - to_frame) < (1.5 * freq):
                        GLOBAL_sst_right.append((next_ic - to_frame) / freq)
        
            # --- Stance Time ---
            # GAUCHE
            for ic_frame in l_starts:
                next_tos = [to for to in lhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        GLOBAL_stance_left.append((to_frame - ic_frame) / freq)
            
            # DROITE
            for ic_frame in r_starts:
                next_tos = [to for to in rhee_toe_offs if to > ic_frame]
                if next_tos:
                    to_frame = next_tos[0]
                    if (to_frame - ic_frame) < (2.0 * freq):
                        GLOBAL_stance_right.append((to_frame - ic_frame) / freq)
        
        # Fin de la boucle sur les fichiers
        print("Traitement de tous les fichiers terminé.")
        
        # ==============================================================================
        # CONVERSION EN ARRAYS NUMPY (Données consolidées)
        # ==============================================================================
        
        # Step Length
        sl_left_arr = np.array(GLOBAL_sl_left)
        sl_right_arr = np.array(GLOBAL_sl_right)
        
        # Step Time
        st_left_arr = np.array(GLOBAL_st_left)
        st_right_arr = np.array(GLOBAL_st_right)
        
        # Velocity
        sv_left_arr = np.array(GLOBAL_sv_left)
        sv_right_arr = np.array(GLOBAL_sv_right)
        
        # Supports
        sst_left = np.array(GLOBAL_sst_left)
        sst_right = np.array(GLOBAL_sst_right)
        dst_left = np.array(GLOBAL_dst_left)
        dst_right = np.array(GLOBAL_dst_right)
        stance_time_left = np.array(GLOBAL_stance_left)
        stance_time_right = np.array(GLOBAL_stance_right)
        
        
        # ==============================================================================
        # CALCUL DES PARAMÈTRES EGVI (Moyennes, Normalisation, Différences)
        # ==============================================================================
        
        def calculate_egvi_components(data_array, name, is_length_cm_conversion=False):
            """
            Fonction générique pour calculer la moyenne et la variabilité (SD Diff)
            selon la logique stricte du script original.
            """
            if len(data_array) == 0:
                return np.nan, np.nan, np.nan # Mean, MeanDiff, StdDiff
        
            # 1. Calcul Moyenne brute (pour normalisation)
            # Attention: pour step length le script original divise par 10 (mm->cm ?) à l'affichage
            # Mais pour l'EGVI la normalisation se fait par la moyenne du dataset
            mean_val = np.mean(data_array)
        
            # 2. Normalisation (P = val / mean)
            P_list = []
            for val in data_array:
                P_list.append(val / mean_val)
            
            # 3. Différences successives (|Pn+1 - Pn|)
            Diff_list = []
            for i in range(len(P_list) - 1):
                valeur = abs(P_list[i+1] - P_list[i])
                Diff_list.append(valeur)
            
            # 4. Stats sur les différences
            if len(Diff_list) > 0:
                mean_diff = np.mean(Diff_list)
                std_diff = np.std(Diff_list, ddof=1)
            else:
                mean_diff = np.nan
                std_diff = np.nan
                
            return mean_val, mean_diff, std_diff
        
        # --- APPLIQUER LES CALCULS ---
        
        # 1. Step Length (Attention unité mm dans c3d -> division par 10 si besoin de cm pour l'affichage, mais EGVI use ratios)
        # Dans le script original, "mean_l" servait à l'affichage (/10).
        # Ici on stocke les valeurs normalisées pour l'EGVI.
        
        # GAUCHE
        raw_mean_sl_l, mean_StepLenDiff_left, std_StepLenDiff_left = calculate_egvi_components(sl_left_arr, "SL Left")
        # DROITE
        raw_mean_sl_r, mean_StepLenDiff_right, std_StepLenDiff_right = calculate_egvi_components(sl_right_arr, "SL Right")
        
        # 2. Step Time
        # GAUCHE
        raw_mean_st_l, mean_StepTimeDiff_left, std_StepTimeDiff_left = calculate_egvi_components(st_left_arr, "ST Left")
        # DROITE
        raw_mean_st_r, mean_StepTimeDiff_right, std_StepTimeDiff_right = calculate_egvi_components(st_right_arr, "ST Right")
        
        # 3. Velocity
        # GAUCHE
        raw_mean_sv_l, mean_StrideVelocityDiff_left, std_StrideVelocityDiff_left = calculate_egvi_components(sv_left_arr, "Vel Left")
        # DROITE
        raw_mean_sv_r, mean_StrideVelocityDiff_right, std_StrideVelocityDiff_right = calculate_egvi_components(sv_right_arr, "Vel Right")
        
        # 4. Single Support
        # GAUCHE
        raw_mean_ss_l, mean_SingleSupDiff_left, std_SingleSupDiff_left = calculate_egvi_components(sst_left, "SST Left")
        # DROITE
        raw_mean_ss_r, mean_SingleSupDiff_right, std_SingleSupDiff_right = calculate_egvi_components(sst_right, "SST Right")
        
        # 5. Stance Time
        # GAUCHE
        raw_mean_stance_l, mean_StanceTimeDiff_left, std_StanceTimeDiff_left = calculate_egvi_components(stance_time_left, "Stance Left")
        # DROITE
        raw_mean_stance_r, mean_StanceTimeDiff_right, std_StanceTimeDiff_right = calculate_egvi_components(stance_time_right, "Stance Right")
        
        
        # ==============================================================================
        # CALCUL FINAL EGVI
        # ==============================================================================
        
        def calculer_egvi(donnees_sujet):
            # 1. Coefficients (cn) extraits du fichier GVI calculation.csv
            coeffs = np.array([
                0.80,  # m step l
                0.93,  # m step t
                0.92,  # m stance
                0.90,  # m single
                0.89,  # m velo
                0.73,  # sd step l
                0.82,  # sd step t
                0.85,  # sd stance
                0.86,  # sd single
                0.90   # sd velo
            ])
        
            # 2. Paramètres de référence du groupe de Contrôle
            mean_control_s_alpha = 20.37541132570365
            mean_ln_d_control = 1.3865728033714733
            sd_ln_d_control = 0.6193340454665202
        
            # Validation
            if len(donnees_sujet) != 10:
                print("Erreur : Pas assez de données pour le calcul (nan detecté ?)")
                return np.nan
        
            if np.isnan(donnees_sujet).any():
                print("Attention : Des valeurs NaN sont présentes dans les données sujet.")
                return np.nan
        
            # Calcul
            s_alpha_sujet = np.dot(coeffs, donnees_sujet)
            diff = s_alpha_sujet - mean_control_s_alpha
            d = abs(diff) + 1
            ln_d = math.log(d)
            z_score = (ln_d - mean_ln_d_control) / sd_ln_d_control
            egvi = 100 + (10 * z_score)
        
            return egvi
        
        # Assemblage des vecteurs pour l'EGVI
        
        valeurs_sujet_D = [
            mean_StepLenDiff_right,   # m step l
            mean_StepTimeDiff_right,  # m step t
            mean_StanceTimeDiff_right, # m stance
            mean_SingleSupDiff_right,  # m single
            mean_StrideVelocityDiff_right, # m velo
            std_StepLenDiff_right,    # sd step l
            std_StepTimeDiff_right,   # sd step t
            std_StanceTimeDiff_right,  # sd stance
            std_SingleSupDiff_right,   # sd single
            std_StrideVelocityDiff_right   # sd velo
        ]
        
        valeurs_sujet_G = [
            mean_StepLenDiff_left,   # m step l
            mean_StepTimeDiff_left,  # m step t
            mean_StanceTimeDiff_left, # m stance
            mean_SingleSupDiff_left,  # m single
            mean_StrideVelocityDiff_left, # m velo
            std_StepLenDiff_left,    # sd step l
            std_StepTimeDiff_left,   # sd step t
            std_StanceTimeDiff_left,  # sd stance
            std_SingleSupDiff_left,   # sd single
            std_StrideVelocityDiff_left   # sd velo
        ]
        
        egvi_resultat_D = calculer_egvi(valeurs_sujet_D)
        egvi_resultat_G = calculer_egvi(valeurs_sujet_G)
        EGVItot = (egvi_resultat_D + egvi_resultat_G)/2
        
        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"Nombre total de cycles (Approx) : {len(sl_left_arr) + len(sl_right_arr)}")
        st.write(f"**Score eGVI** : {EGVItot:.2f}")
        st.write(f"**Score eGVI droit** : {egvi_resultat_D:.2f}")
        st.write(f"**Score eGVI gauche** : {egvi_resultat_G:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 98 et 102. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
       
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")

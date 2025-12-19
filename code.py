import streamlit as st
import ezc3d
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
        # 1. INITIALISATION DES LISTES GLOBALES
        Step_lenght_D = []
        Step_time_D = []
        Double_Supp_time_D = []
        Single_Supp_time_D = []
        Stride_length_D = []
        Stride_time_D = []
        Stride_velocity_D = []
        Stance_Time_D = []
        
        files = [tmp1_path, tmp2_path, tmp3_path]
        FS = 100
        AXE_Y = 1  # Progression
        AXE_Z = 2  # Hauteur
        
        # --- BOUCLE PRINCIPALE SUR LES FICHIERS ---
        for filename in files:
            try:
                # Chargement du fichier
                acq = ezc3d.c3d(filename)
                data = acq['data']['points']
                labels = acq['parameters']['POINT']['LABELS']['value']
        
                # Récupération des indices
                idx_RHEE = labels.index('RHEE')
                idx_LHEE = labels.index('LHEE')
                idx_RTOE = labels.index('RTOE')
                idx_LTOE = labels.index('LTOE')
        
                max_f = data.shape[2]
            except Exception as e:
                print(f"Erreur sur le fichier {filename}: {e}")
                continue
        
            # --- DÉTECTION DES CYCLES ---
            # Détection des Heel Strikes (HS) par les minima de Z (inversés en pics)
            z_rhee = data[AXE_Z, idx_RHEE, :]
            z_lhee = data[AXE_Z, idx_LHEE, :]
            z_rtoe = data[AXE_Z, idx_RTOE, :]
            z_ltoe = data[AXE_Z, idx_LTOE, :]
        
            rhs_peaks, _ = find_peaks(-z_rhee, distance=50, prominence=10)
            lhs_peaks, _ = find_peaks(-z_lhee, distance=50, prominence=10)
            rto_peaks, _ = find_peaks(-z_rtoe, distance=50, prominence=10)
            lto_peaks, _ = find_peaks(-z_ltoe, distance=50, prominence=10)
        
            # Conversion au format attendu par votre script :
            # r_cycles : [(HS_i, HS_j), ...]
            r_cycles = [(rhs_peaks[k], rhs_peaks[k+1]) for k in range(len(rhs_peaks)-1)]
            # l_cycles : [(HS_i, HS_j), ...] (Formaté pour le Single Support)
            l_cycles = [(lhs_peaks[k], lhs_peaks[k+1]) for k in range(len(lhs_peaks)-1)]
            # rt_cycles : [(HS, TO), ...]
            # On apparie chaque HS droit avec son TO droit suivant
            rt_cycles = []
            for hs in rhs_peaks:
                future_to = rto_peaks[rto_peaks > hs]
                if len(future_to) > 0:
                    rt_cycles.append((hs, future_to[0]))
        
            # Fonction pour l'accès aux données (Axe, Marqueur, Frame)
            def get_y(frame, m_idx):
                if frame < max_f:
                    return data[AXE_Y, m_idx, frame]
                return None
        
            # --- A. STRIDES (Enjambées) ---
            for i, j in r_cycles:
                p_i = get_y(i, idx_RHEE)
                p_j = get_y(j, idx_RHEE)
                if p_i is not None and p_j is not None:
                    dist = np.abs(p_j - p_i)
                    dur = (j - i) / FS
                    Stride_length_D.append(dist)
                    Stride_time_D.append(dur)
                    Stride_velocity_D.append(dist / dur)
        
            # --- B. STEPS & SINGLE SUPPORT (Pas et Support Simple) ---
            nb = min(len(r_cycles), len(l_cycles))
            for i in range(nb):
                rhs = r_cycles[i][0]
                lhs = l_cycles[i][0]
                pR, pL = get_y(rhs, idx_RHEE), get_y(lhs, idx_LHEE)
        
                if pR is not None and pL is not None:
                    if lhs < rhs:
                        Step_time_D.append((rhs - lhs) / FS)
                        Step_lenght_D.append(np.abs(pR - pL))
                        if i + 1 < len(l_cycles):
                            Single_Supp_time_D.append((l_cycles[i+1][0] - rhs) / FS)
                    else:
                        if i + 1 < len(r_cycles):
                            n_rhs = r_cycles[i+1][0]
                            Step_time_D.append((n_rhs - lhs) / FS)
                            pnR = get_y(n_rhs, idx_RHEE)
                            if pnR is not None:
                                Step_lenght_D.append(np.abs(pnR - pL))
                        Single_Supp_time_D.append((lhs - rhs) / FS)
        
            # --- C. DOUBLE SUPPORT ---
            nb_ds = min(len(l_cycles), len(rt_cycles))
            for i in range(nb_ds):
                lhs_hs, rto = l_cycles[i][0], rt_cycles[i][1]
                if lhs_hs < max_f and rto < max_f:
                    Double_Supp_time_D.append(max(0, (rto - lhs_hs) / FS))
        
            # Stance time
            for i in range(len(Single_Supp_time_D)):
              Stance_Time_D.append(Single_Supp_time_D[i] + Double_Supp_time_D[i])
            # Valeur moyenne
            m_Stride_velocity_D = np.mean(Stride_velocity_D)
            m_Step_lenght_D = np.mean(Step_lenght_D)
            m_Step_time_D = np.mean(Step_time_D)
            m_Single_Supp_time_D = np.mean(Single_Supp_time_D)
            m_Stance_Time_D = np.mean(Stance_Time_D)
            
            # En pourcentage de la moyenne
            
            pm_Stride_velocity_D = []
            pm_Step_lenght_D = []
            pm_Step_time_D = []
            pm_Single_Supp_time_D =[]
            pm_Stance_Time_D = []
            
            for i in range(len(Stride_velocity_D)) :
              pm_Stride_velocity_D.append((Stride_velocity_D[i]/m_Stride_velocity_D)*100)
            
            for i in range(len(Step_lenght_D)) :
              pm_Step_lenght_D.append((Step_lenght_D[i]/m_Step_lenght_D)*100)
            
            for i in range(len(Step_time_D)) :
              pm_Step_time_D.append((Step_time_D[i]/m_Step_time_D)*100)
            
            for i in range(len(Single_Supp_time_D)) :
              pm_Single_Supp_time_D.append((Single_Supp_time_D[i]/m_Single_Supp_time_D)*100)
            
            for i in range(len(Stance_Time_D)) :
              pm_Stance_Time_D.append((Stance_Time_D[i]/m_Stance_Time_D)*100)
            #Différence absolue
            abs_Stride_velocity_D = []
            abs_Step_lenght_D = []
            abs_Step_time_D = []
            abs_Single_Supp_time_D =[]
            abs_Stance_Time_D = []
            
            for i in range(len(pm_Stride_velocity_D)-1):
              abs_Stride_velocity_D.append(np.abs(pm_Stride_velocity_D[i+1]-pm_Stride_velocity_D[i]))
            
            for i in range(len(pm_Step_lenght_D)-1):
              abs_Step_lenght_D.append(np.abs(pm_Step_lenght_D[i+1]-pm_Step_lenght_D[i]))
            
            for i in range(len(pm_Step_time_D)-1):
              abs_Step_time_D.append(np.abs(pm_Step_time_D[i+1]-pm_Step_time_D[i]))
            
            for i in range(len(pm_Single_Supp_time_D)-1):
              abs_Single_Supp_time_D.append(np.abs(pm_Single_Supp_time_D[i+1]-pm_Single_Supp_time_D[i]))
            
            for i in range(len(pm_Stance_Time_D)-1):
              abs_Stance_Time_D.append(np.abs(pm_Stance_Time_D[i+1]-pm_Stance_Time_D[i]))
            
            # Moyenne des différences absolues
            m_abs_Stride_velocity_D = np.mean(abs_Stride_velocity_D)
            m_abs_Step_lenght_D = np.mean(abs_Step_lenght_D)
            m_abs_Step_time_D = np.mean(abs_Step_time_D)
            m_abs_Single_Supp_time_D = np.mean(abs_Single_Supp_time_D)
            m_abs_Stance_Time_D = np.mean(abs_Stance_Time_D)
            
            # Ecart-type des différences absolues
            et_abs_Stride_velocity_D = np.std(abs_Stride_velocity_D)
            et_abs_Step_lenght_D = np.std(abs_Step_lenght_D)
            et_abs_Step_time_D = np.std(abs_Step_time_D)
            et_abs_Single_Supp_time_D = np.std(abs_Single_Supp_time_D)
            et_abs_Stance_Time_D = np.std(abs_Stance_Time_D)
            
            # 1. INITIALISATION DES LISTES GLOBALES (Côté GAUCHE)
            Step_lenght_G = []
            Step_time_G = []
            Double_Supp_time_G = []
            Single_Supp_time_G = []
            Stride_length_G = []
            Stride_time_G = []
            Stride_velocity_G = []
            Stance_Time_G = []
            
            files = ["/BureauR03.c3d", "/BureauR04.c3d", "/BureauR05.c3d"]
            FS = 100
            AXE_Y = 1  # Progression
            AXE_Z = 2  # Hauteur
            
            # --- BOUCLE PRINCIPALE SUR LES FICHIERS ---
            for filename in files:
                try:
                    # Chargement du fichier
                    acq = ezc3d.c3d(filename)
                    data = acq['data']['points']
                    labels = acq['parameters']['POINT']['LABELS']['value']
            
                    # Récupération des indices
                    idx_RHEE = labels.index('RHEE')
                    idx_LHEE = labels.index('LHEE')
                    idx_RTOE = labels.index('RTOE')
                    idx_LTOE = labels.index('LTOE')
            
                    max_f = data.shape[2]
                except Exception as e:
                    print(f"Erreur sur le fichier {filename}: {e}")
                    continue
            
            # --- DÉTECTION DES CYCLES ---
            z_rhee = data[AXE_Z, idx_RHEE, :]
            z_lhee = data[AXE_Z, idx_LHEE, :]
            z_ltoe = data[AXE_Z, idx_LTOE, :]
        
            rhs_peaks, _ = find_peaks(-z_rhee, distance=50, prominence=10)
            lhs_peaks, _ = find_peaks(-z_lhee, distance=50, prominence=10)
            lto_peaks, _ = find_peaks(-z_ltoe, distance=50, prominence=10)
        
            # r_cycles : [(HS_i, HS_j), ...]
            r_cycles = [(rhs_peaks[k], rhs_peaks[k+1]) for k in range(len(rhs_peaks)-1)]
            # l_cycles : [(HS_i, HS_j), ...]
            l_cycles = [(lhs_peaks[k], lhs_peaks[k+1]) for k in range(len(lhs_peaks)-1)]
        
            # lt_cycles : [(HS_G, TO_G), ...] pour le pied gauche
            lt_cycles = []
            for hs in lhs_peaks:
                future_to = lto_peaks[lto_peaks > hs]
                if len(future_to) > 0:
                    lt_cycles.append((hs, future_to[0]))
        
            def get_y(frame, m_idx):
                if frame < max_f:
                    return data[AXE_Y, m_idx, frame]
                return None
        
            # --- A. STRIDES (Enjambées Gauches) ---
            for i, j in l_cycles:
                p_i = get_y(i, idx_LHEE)
                p_j = get_y(j, idx_LHEE)
                if p_i is not None and p_j is not None:
                    dist = np.abs(p_j - p_i)
                    dur = (j - i) / FS
                    Stride_length_G.append(dist)
                    Stride_time_G.append(dur)
                    Stride_velocity_G.append(dist / dur)
        
            # --- B. STEPS & SINGLE SUPPORT (Côté Gauche) ---
            # Le step gauche se mesure par rapport au contact droit précédent
            nb = min(len(r_cycles), len(l_cycles))
            for i in range(nb):
                rhs = r_cycles[i][0]
                lhs = l_cycles[i][0]
                pR, pL = get_y(rhs, idx_RHEE), get_y(lhs, idx_LHEE)
        
                if pR is not None and pL is not None:
                    # Si le contact droit est avant le gauche
                    if rhs < lhs:
                        Step_time_G.append((lhs - rhs) / FS)
                        Step_lenght_G.append(np.abs(pL - pR))
                        if i + 1 < len(r_cycles):
                            # Single Support G : pied droit en l'air (HS G -> HS D suivant)
                            Single_Supp_time_G.append((r_cycles[i+1][0] - lhs) / FS)
                    else:
                        if i + 1 < len(l_cycles):
                            n_lhs = l_cycles[i+1][0]
                            Step_time_G.append((n_lhs - rhs) / FS)
                            pnL = get_y(n_lhs, idx_LHEE)
                            if pnL is not None:
                                Step_lenght_G.append(np.abs(pnL - pR))
                        Single_Supp_time_G.append((rhs - lhs) / FS)
        
            # --- C. DOUBLE SUPPORT (G) ---
            # Temps entre l'impact droit (RHS) et le décollage gauche (LTO)
            nb_ds = min(len(r_cycles), len(lt_cycles))
            for i in range(nb_ds):
                rhs_hs = r_cycles[i][0]
                lto = lt_cycles[i][1]
                if rhs_hs < max_f and lto < max_f:
                    Double_Supp_time_G.append(max(0, (lto - rhs_hs) / FS))
        
            # --- D. STANCE TIME (G) ---
            # Somme du support simple gauche et du double support associé
            temp_nb = min(len(Single_Supp_time_G), len(Double_Supp_time_G))
            # On vide et recalcule Stance pour chaque fichier pour garder la cohérence du cumul global
            # ou on peut simplement additionner les derniers éléments ajoutés
            start_idx = len(Stance_Time_G)
            for i in range(start_idx, temp_nb):
                Stance_Time_G.append(Single_Supp_time_G[i] + Double_Supp_time_G[i])
            # Valeur moyenne GAUCHE
            m_Stride_velocity_G = np.mean(Stride_velocity_G)
            m_Step_lenght_G = np.mean(Step_lenght_G)
            m_Step_time_G = np.mean(Step_time_G)
            m_Single_Supp_time_G = np.mean(Single_Supp_time_G)
            m_Stance_Time_G = np.mean(Stance_Time_G)
            
            # En pourcentage de la moyenne GAUCHE
            pm_Stride_velocity_G = []
            pm_Step_lenght_G = []
            pm_Step_time_G = []
            pm_Single_Supp_time_G = []
            pm_Stance_Time_G = []
            
            for i in range(len(Stride_velocity_G)):
                pm_Stride_velocity_G.append((Stride_velocity_G[i] / m_Stride_velocity_G) * 100)
            
            for i in range(len(Step_lenght_G)):
                pm_Step_lenght_G.append((Step_lenght_G[i] / m_Step_lenght_G) * 100)
            
            for i in range(len(Step_time_G)):
                pm_Step_time_G.append((Step_time_G[i] / m_Step_time_G) * 100)
            
            for i in range(len(Single_Supp_time_G)):
                pm_Single_Supp_time_G.append((Single_Supp_time_G[i] / m_Single_Supp_time_G) * 100)
            
            for i in range(len(Stance_Time_G)):
                pm_Stance_Time_G.append((Stance_Time_G[i] / m_Stance_Time_G) * 100)
            # Différence absolue GAUCHE
            abs_Stride_velocity_G = []
            abs_Step_lenght_G = []
            abs_Step_time_G = []
            abs_Single_Supp_time_G = []
            abs_Stance_Time_G = []
            
            # Calcul des différences entre cycles successifs (i+1 et i)
            for i in range(len(pm_Stride_velocity_G)-1):
                abs_Stride_velocity_G.append(np.abs(pm_Stride_velocity_G[i+1] - pm_Stride_velocity_G[i]))
            
            for i in range(len(pm_Step_lenght_G)-1):
                abs_Step_lenght_G.append(np.abs(pm_Step_lenght_G[i+1] - pm_Step_lenght_G[i]))
            
            for i in range(len(pm_Step_time_G)-1):
                abs_Step_time_G.append(np.abs(pm_Step_time_G[i+1] - pm_Step_time_G[i]))
            
            for i in range(len(pm_Single_Supp_time_G)-1):
                abs_Single_Supp_time_G.append(np.abs(pm_Single_Supp_time_G[i+1] - pm_Single_Supp_time_G[i]))
            
            for i in range(len(pm_Stance_Time_G)-1):
                abs_Stance_Time_G.append(np.abs(pm_Stance_Time_G[i+1] - pm_Stance_Time_G[i]))
            
            # Moyenne des différences absolues GAUCHE
            m_abs_Stride_velocity_G = np.mean(abs_Stride_velocity_G)
            m_abs_Step_lenght_G = np.mean(abs_Step_lenght_G)
            m_abs_Step_time_G = np.mean(abs_Step_time_G)
            m_abs_Single_Supp_time_G = np.mean(abs_Single_Supp_time_G)
            m_abs_Stance_Time_G = np.mean(abs_Stance_Time_G)
            
            # Ecart-type des différences absolues GAUCHE
            et_abs_Stride_velocity_G = np.std(abs_Stride_velocity_G)
            et_abs_Step_lenght_G = np.std(abs_Step_lenght_G)
            et_abs_Step_time_G = np.std(abs_Step_time_G)
            et_abs_Single_Supp_time_G = np.std(abs_Single_Supp_time_G)
            et_abs_Stance_Time_G = np.std(abs_Stance_Time_G)
            # Création des vecteurs droit et gauche
            Vect_D = [m_abs_Step_lenght_D, m_abs_Step_time_D, m_abs_Stance_Time_D, m_abs_Single_Supp_time_D, m_abs_Stride_velocity_D, et_abs_Step_lenght_D, et_abs_Step_time_D, et_abs_Stance_Time_D, et_abs_Single_Supp_time_D, et_abs_Stride_velocity_D]
            Vect_G = [m_abs_Step_lenght_G, m_abs_Step_time_G, m_abs_Stance_Time_G, m_abs_Single_Supp_time_G, m_abs_Stride_velocity_G, et_abs_Step_lenght_G, et_abs_Step_time_G, et_abs_Stance_Time_G, et_abs_Single_Supp_time_G, et_abs_Stride_velocity_G]
            
            # Réalisation des somprod et obtention de la projection du sujet
            SP_D = np.sum(Vect_D * cn, axis=0)
            SP_G = np.sum(Vect_G * cn, axis=0)
            
            Diff_Sujet_CTRL_G = (SP_G-Projection_CTRL)
            Diff_Sujet_CTRL_G2 = abs(Diff_Sujet_CTRL_G)+1
            Diff_Sujet_CTRL_D = (SP_D-Projection_CTRL);
            Diff_Sujet_CTRL_D2 = abs(Diff_Sujet_CTRL_D)+1
            
            # Fin et calcul eGVI
            if Diff_Sujet_CTRL_G < 0 :
                ln_sujet_G = -np.log(Diff_Sujet_CTRL_G2)
            else :
                ln_sujet_G = np.log(Diff_Sujet_CTRL_G2)
            
            if Diff_Sujet_CTRL_D < 0 :
                ln_sujet_D = -np.log(Diff_Sujet_CTRL_D2)
            else :
                ln_sujet_D = np.log(Diff_Sujet_CTRL_D2)
            
            # z gauche
            z_G = []
            
            if ln_sujet_G < -m_ln_CTRL :
                z_G = (ln_sujet_G + m_ln_CTRL) / sd_ln_CTRL
            
            if ln_sujet_G > m_ln_CTRL :
                z_G = (ln_sujet_G - m_ln_CTRL) / sd_ln_CTRL
            
            if -m_ln_CTRL < ln_sujet_G and ln_sujet_G < m_ln_CTRL :
                z_G = 0
            
            # z droit
            z_D =  []
            
            if ln_sujet_D < -m_ln_CTRL :
                z_D = (ln_sujet_D + m_ln_CTRL) / sd_ln_CTRL
            
            if ln_sujet_D > m_ln_CTRL :
                z_D = (ln_sujet_D - m_ln_CTRL) / sd_ln_CTRL
            
            if -m_ln_CTRL < ln_sujet_D and  ln_sujet_D < m_ln_CTRL :
                z_D = 0
            
            eGVI_G = 100+z_G
            eGVI_D = 100+z_D
            eGVI = (eGVI_D + eGVI_G)/2

            st.markdown("### 📊 Résultats du score eGVI")
            st.write(f"**Score eGVI** : {eGVI:.2f}")
            st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 98 et 102. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
       
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")

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
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique1 = st.selectbox("Choisissez un fichier dynamique 1 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique2 = st.selectbox("Choisissez un fichier dynamique 2 pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique3 = st.selectbox("Choisissez un fichier dynamique 3 pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique1.read())
        tmp1_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique2.read())
        tmp2_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique3.read())
        tmp3_path = tmp.name

    acq1 = ezc3d.c3d(tmp1_path)  # acquisition dynamique
    labels1 = acq1['parameters']['POINT']['LABELS']['value']
    freq1 = acq1['header']['points']['frame_rate']
    first_frame1 = acq1['header']['points']['first_frame']
    n_frames1 = acq1['data']['points'].shape[2]
    time_offset1 = first_frame1 / freq1
    time1 = np.arange(n_frames1) / freq1 + time_offset1
    error_acq1 = False

    acq2 = ezc3d.c3d(tmp2_path)  # acquisition dynamique
    labels2 = acq2['parameters']['POINT']['LABELS']['value']
    freq2 = acq2['header']['points']['frame_rate']
    first_frame2 = acq2['header']['points']['first_frame']
    n_frames2 = acq2['data']['points'].shape[2]
    time_offset2 = first_frame2 / freq2
    time2 = np.arange(n_frames2) / freq2 + time_offset2
    error_acq2 = False

    acq3 = ezc3d.c3d(tmp3_path)  # acquisition dynamique
    labels3 = acq3['parameters']['POINT']['LABELS']['value']
    freq3 = acq3['header']['points']['frame_rate']
    first_frame3 = acq3['header']['points']['first_frame']
    n_frames3 = acq3['data']['points'].shape[2]
    time_offset3 = first_frame3 / freq3
    time3 = np.arange(n_frames3) / freq3 + time_offset3
    error_acq3 = False
    
    statique = ezc3d.c3d(tmp_path)  # acquisition statique
    labelsStat = statique['parameters']['POINT']['LABELS']['value']
    freqStat = statique['header']['points']['frame_rate']
    first_frameStat = statique['header']['points']['first_frame']
    n_framesStat = statique['data']['points'].shape[2]
    time_offsetStat = first_frameStat / freqStat
    timeStat = np.arange(n_framesStat) / freqStat + time_offsetStat
    
    markersStat  = statique['data']['points']
    markers1 = acq1['data']['points']
    markers2 = acq2['data']['points']
    markers3 = acq3['data']['points']
    data1 = acq1['data']['points']
    data2 = acq2['data']['points']
    data3 = acq3['data']['points']

    # Valeur population contrôle Arnaud Gouelle
    cn = [0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90]
    cn = np.array(cn)
    m_ln_CTRL = 1.386573
    sd_ln_CTRL = 0.619334
    Projection_CTRL = 20.38

if st.button("Lancer le calcul du score eGVI"):
    try:
         # Extraction des coordonnées
        a1, a2, b1, b2, c1, c2 = markersStat[:,labelsStat.index('LASI'),:][0, 0], markersStat[:,labelsStat.index('LANK'),:][0, 0], markersStat[:,labelsStat.index('LASI'),:][1, 0], markersStat[:,labelsStat.index('LANK'),:][1, 0], markersStat[:,labelsStat.index('LASI'),:][2, 0], markersStat[:,labelsStat.index('LANK'),:][2, 0]
        LgJambeL = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))

        d1, d2, e1, e2, f1, f2 = markersStat[:,labelsStat.index('RASI'),:][0, 0], markersStat[:,labelsStat.index('RANK'),:][0, 0], markersStat[:,labelsStat.index('RASI'),:][1, 0], markersStat[:,labelsStat.index('RANK'),:][1, 0], markersStat[:,labelsStat.index('RASI'),:][2, 0], markersStat[:,labelsStat.index('RANK'),:][2, 0]
        LgJambeR = np.sqrt((d2-d1)*(d2-d1)+(e2-e1)*(e2-e1)+(f2-f1)*(f2-f1))
      
        # Cycles première acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points1 = acq1['data']['points']
        if "LHEE" in labels1:
            idx_lhee1 = labels1.index("LHEE")
            z_lhee1 = points1[2, idx_lhee1, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z1 = -z_lhee1
            min_distance1 = int(freq1 * 0.8)
        
            # Détection pics
            peaks1, _ = find_peaks(inverted_z1, distance = min_distance1, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices1 = peaks1[:-1]
            lhee_cycle_end_indices1 = peaks1[1:]
            min_lhee_cycle_duration1 = int(0.5 * freq1)
            lhee_valid_cycles1 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices1, lhee_cycle_end_indices1)
              if (end - start) >= min_lhee_cycle_duration1
            ]
            lhee_n_cycles1 = len(lhee_valid_cycles1)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points1 = acq1['data']['points']
        if "RHEE" in labels1:
            idx_rhee1 = labels1.index("RHEE")
            z_rhee1 = points1[2, idx_rhee1, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z1 = -z_rhee1
            min_distance1 = int(freq1 * 0.8)
        
            # Détection pics
            peaks1, _ = find_peaks(inverted_z1, distance = min_distance1, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices1 = peaks1[:-1]
            rhee_cycle_end_indices1 = peaks1[1:]
            min_rhee_cycle_duration1 = int(0.5 * freq1)
            rhee_valid_cycles1 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices1, rhee_cycle_end_indices1)
              if (end - start) >= min_rhee_cycle_duration1
            ]
            rhee_n_cycles1 = len(rhee_valid_cycles1)
        # Cycles deuxième acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points2 = acq2['data']['points']
        if "LHEE" in labels2:
            idx_lhee2 = labels2.index("LHEE")
            z_lhee2 = points2[2, idx_lhee2, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z2 = -z_lhee2
            min_distance2 = int(freq2 * 0.8)
        
            # Détection pics
            peaks2, _ = find_peaks(inverted_z2, distance = min_distance2, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices2 = peaks2[:-1]
            lhee_cycle_end_indices2 = peaks2[1:]
            min_lhee_cycle_duration2 = int(0.5 * freq2)
            lhee_valid_cycles2 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices2, lhee_cycle_end_indices2)
              if (end - start) >= min_lhee_cycle_duration2
            ]
            lhee_n_cycles2 = len(lhee_valid_cycles2)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points2 = acq2['data']['points']
        if "RHEE" in labels2:
            idx_rhee2 = labels2.index("RHEE")
            z_rhee2 = points2[2, idx_rhee2, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z2 = -z_rhee2
            min_distance2 = int(freq2 * 0.8)
        
            # Détection pics
            peaks2, _ = find_peaks(inverted_z2, distance = min_distance2, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices2 = peaks2[:-1]
            rhee_cycle_end_indices2 = peaks2[1:]
            min_rhee_cycle_duration2 = int(0.5 * freq2)
            rhee_valid_cycles2 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices2, rhee_cycle_end_indices2)
              if (end - start) >= min_rhee_cycle_duration2
            ]
            rhee_n_cycles2 = len(rhee_valid_cycles2)
        # Cycles troisième acquisition
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points3 = acq3['data']['points']
        if "LHEE" in labels3:
            idx_lhee3 = labels3.index("LHEE")
            z_lhee3 = points3[2, idx_lhee3, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z3 = -z_lhee3
            min_distance3 = int(freq3 * 0.8)
        
            # Détection pics
            peaks3, _ = find_peaks(inverted_z3, distance = min_distance3, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices3 = peaks3[:-1]
            lhee_cycle_end_indices3 = peaks3[1:]
            min_lhee_cycle_duration3 = int(0.5 * freq3)
            lhee_valid_cycles3 = [
              (start, end) for start, end in zip(lhee_cycle_start_indices3, lhee_cycle_end_indices3)
              if (end - start) >= min_lhee_cycle_duration3
            ]
            lhee_n_cycles3 = len(lhee_valid_cycles3)
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points3 = acq3['data']['points']
        if "RHEE" in labels3:
            idx_rhee3 = labels3.index("RHEE")
            z_rhee3 = points3[2, idx_rhee3, :]
        
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z3 = -z_rhee3
            min_distance3 = int(freq3 * 0.8)
        
            # Détection pics
            peaks3, _ = find_peaks(inverted_z3, distance = min_distance3, prominence = 1)
        
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices3 = peaks3[:-1]
            rhee_cycle_end_indices3 = peaks3[1:]
            min_rhee_cycle_duration3 = int(0.5 * freq3)
            rhee_valid_cycles3 = [
              (start, end) for start, end in zip(rhee_cycle_start_indices3, rhee_cycle_end_indices3)
              if (end - start) >= min_rhee_cycle_duration3
            ]
            rhee_n_cycles3 = len(rhee_valid_cycles3)

        # COTE DROIT
        PA_D1 = []
        SA_D1 = []
        LPas_D1 = []
        Vitesse_D1 = []
        DPas_D1 = []
        
        for i,j in rhee_valid_cycles1 :
          PA_D1.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_D1.append(np.abs((data1[:,labels1.index('RTOE'),:][1, n] - data1[:,labels1.index('LTOE'),:][1, n])/10))
          Vitesse_D1.append(np.abs(((data1[:,labels1.index('RTOE'),:][1, n] - data1[:,labels1.index('RTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in lhee_valid_cycles1 :
            DPas_D1.append(np.abs(k - i)/100)
            SA_D1.append(np.abs(k-m))
        
        if len(DPas_D1) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_D1.pop(i)
            SA_D1.pop(i)
        
        elif len(DPas_D1) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_D1.pop(i)
            SA_D1.pop(i)
        
        elif len(DPas_D1) == 3:
          error_acq1 = True
        
          # COTE GAUCHE
        PA_G1 = []
        SA_G1 = []
        LPas_G1 = []
        Vitesse_G1 = []
        DPas_G1 = []
        
        for i,j in lhee_valid_cycles1 :
          PA_G1.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_G1.append(np.abs((data1[:,labels1.index('LTOE'),:][1, n] - data1[:,labels1.index('RTOE'),:][1, n])/10))
          Vitesse_G1.append(np.abs(((data1[:,labels1.index('LTOE'),:][1, n] - data1[:,labels1.index('LTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in rhee_valid_cycles1 :
            DPas_G1.append(np.abs(k - i)/100)
            SA_G1.append(np.abs(k-m))
        
        if len(DPas_G1) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_G1.pop(i)
            SA_G1.pop(i)
        
        elif len(DPas_G1) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_G1.pop(i)
            SA_G1.pop(i)
        
        elif len(DPas_G1) == 3:
          error_acq1 = True
        
        if error_acq1 == False:
          # Valeur moyenne
          PA_D_m1 = np.mean(PA_D1)
          LPas_D_m1 = np.mean(LPas_D1)
          DPas_D_m1 = np.mean(DPas_D1)
          Vitesse_D_m1 = np.mean(Vitesse_D1)
          SA_D_m1 = np.mean(SA_D1)
        
          # En pourcentage de la valeur moyenne
          pPA_Dm1 = []
          pDPas_Dm1 = []
          pLPas_Dm1 = []
          pVitesse_Dm1 = []
          pSA_Dm1 = []
          if len(PA_D1) < len(DPas_D1) : 
            for i in range(0,len(PA_D1),1) :
              pPA_Dm1.append(PA_D1[i] * 100 / PA_D_m1)
              pDPas_Dm1.append(DPas_D1[i] * 100 / DPas_D_m1)
              pLPas_Dm1.append(LPas_D1[i] * 100 / LPas_D_m1)
              pVitesse_Dm1.append(Vitesse_D1[i] * 100 / Vitesse_D_m1)
              pSA_Dm1.append(SA_D1[i] * 100 / SA_D_m1)
          else : 
            for i in range(0,len(DPas_D1),1) :
              pPA_Dm1.append(PA_D1[i] * 100 / PA_D_m1)
              pDPas_Dm1.append(DPas_D1[i] * 100 / DPas_D_m1)
              pLPas_Dm1.append(LPas_D1[i] * 100 / LPas_D_m1)
              pVitesse_Dm1.append(Vitesse_D1[i] * 100 / Vitesse_D_m1)
              pSA_Dm1.append(SA_D1[i] * 100 / SA_D_m1)
        
          # Différence absolue
          PA_D_f1 = []
          DPas_D_f1 = []
          LPas_D_f1 = []
          Vitesse_D_f1 = []
          SA_D_f1 = []
        
          if len(PA_D1) < len(DPas_D1) :
            for i in range(0,len(PA_D1)-1,1) :
              PA_D_f1.append(np.abs(pPA_Dm1[i+1] - pPA_Dm1[i]))
              DPas_D_f1.append(np.abs(pDPas_Dm1[i+1] - pDPas_Dm1[i]))
              LPas_D_f1.append(np.abs(pLPas_Dm1[i+1] - pLPas_Dm1[i]))
              Vitesse_D_f1.append(np.abs(pVitesse_Dm1[i+1] - pVitesse_Dm1[i]))
              SA_D_f1.append(np.abs(pSA_Dm1[i+1] - pSA_Dm1[i]))
          else : 
            for i in range(0,len(DPas_D1)-1,1) :
              PA_D_f1.append(np.abs(pPA_Dm1[i+1] - pPA_Dm1[i]))
              DPas_D_f1.append(np.abs(pDPas_Dm1[i+1] - pDPas_Dm1[i]))
              LPas_D_f1.append(np.abs(pLPas_Dm1[i+1] - pLPas_Dm1[i]))
              Vitesse_D_f1.append(np.abs(pVitesse_Dm1[i+1] - pVitesse_Dm1[i]))
              SA_D_f1.append(np.abs(pSA_Dm1[i+1] - pSA_Dm1[i]))
        
          # Valeur moyenne
          PA_G_m1 = np.mean(PA_G1)
          LPas_G_m1 = np.mean(LPas_G1)
          DPas_G_m1 = np.mean(DPas_G1)
          Vitesse_G_m1 = np.mean(Vitesse_G1)
          SA_G_m1 = np.mean(SA_G1)
        
          # En pourcentage de la valeur moyenne
          pPA_Gm1 = []
          pDPas_Gm1 = []
          pLPas_Gm1 = []
          pVitesse_Gm1 = []
          pSA_Gm1 = []
          if len(PA_G1) < len(DPas_G1) : 
            for i in range(0,len(PA_G1),1) :
              pPA_Gm1.append(PA_G1[i] * 100 / PA_G_m1)
              pDPas_Gm1.append(DPas_G1[i] * 100 / DPas_G_m1)
              pLPas_Gm1.append(LPas_G1[i] * 100 / LPas_G_m1)
              pVitesse_Gm1.append(Vitesse_G1[i] * 100 / Vitesse_G_m1)
              pSA_Gm1.append(SA_G1[i] * 100 / SA_G_m1)
          else : 
            for i in range(0,len(DPas_G1),1) :
              pPA_Gm1.append(PA_G1[i] * 100 / PA_G_m1)
              pDPas_Gm1.append(DPas_G1[i] * 100 / DPas_G_m1)
              pLPas_Gm1.append(LPas_G1[i] * 100 / LPas_G_m1)
              pVitesse_Gm1.append(Vitesse_G1[i] * 100 / Vitesse_G_m1)
              pSA_Gm1.append(SA_G1[i] * 100 / SA_G_m1)
        
          # Différence absolue
          PA_G_f1 = []
          DPas_G_f1 = []
          LPas_G_f1 = []
          Vitesse_G_f1 = []
          SA_G_f1 = []
        
          if len(PA_G1) < len(DPas_G1) :
            for i in range(0,len(PA_G1)-1,1) :
              PA_G_f1.append(np.abs(pPA_Gm1[i+1] - pPA_Gm1[i]))
              DPas_G_f1.append(np.abs(pDPas_Gm1[i+1] - pDPas_Gm1[i]))
              LPas_G_f1.append(np.abs(pLPas_Gm1[i+1] - pLPas_Gm1[i]))
              Vitesse_G_f1.append(np.abs(pVitesse_Gm1[i+1] - pVitesse_Gm1[i]))
              SA_G_f1.append(np.abs(pSA_Gm1[i+1] - pSA_Gm1[i]))
          else : 
            for i in range(0,len(DPas_G1)-1,1) :
              PA_G_f1.append(np.abs(pPA_Gm1[i+1] - pPA_Gm1[i]))
              DPas_G_f1.append(np.abs(pDPas_Gm1[i+1] - pDPas_Gm1[i]))
              LPas_G_f1.append(np.abs(pLPas_Gm1[i+1] - pLPas_Gm1[i]))
              Vitesse_G_f1.append(np.abs(pVitesse_Gm1[i+1] - pVitesse_Gm1[i]))
              SA_G_f1.append(np.abs(pSA_Gm1[i+1] - pSA_Gm1[i]))
        
          #Moyenne des différences absolues
          # Droit :
          mean_PA_D_f1 = np.mean(PA_D_f1)
          mean_DPas_D_f1 = np.mean(DPas_D_f1)
          mean_LPas_D_f1 = np.mean(LPas_D_f1)
          mean_Vitesse_D_f1 = np.mean(Vitesse_D_f1)
          mean_SA_D_f1 = np.mean(SA_D_f1)
        
          # Gauche :
          mean_PA_G_f1 = np.mean(PA_G_f1)
          mean_DPas_G_f1 = np.mean(DPas_G_f1)
          mean_LPas_G_f1 = np.mean(LPas_G_f1)
          mean_Vitesse_G_f1 = np.mean(Vitesse_G_f1)
          mean_SA_G_f1 = np.mean(SA_G_f1)
        
          # Ecart-type des différences absolues
          # Droit :
          std_PA_D_f1 = np.std(PA_D_f1)
          std_DPas_D_f1 = np.std(DPas_D_f1)
          std_LPas_D_f1 = np.std(LPas_D_f1)
          std_Vitesse_D_f1 = np.std(Vitesse_D_f1)
          std_SA_D_f1 = np.std(SA_D_f1)
        
          # Gauche :
          std_PA_G_f1 = np.std(PA_G_f1)
          std_DPas_G_f1 = np.std(DPas_G_f1)
          std_LPas_G_f1 = np.std(LPas_G_f1)
          std_Vitesse_G_f1 = np.std(Vitesse_G_f1)
          std_SA_G_f1 = np.std(SA_G_f1)
        
          # Création des vecteurs droit et gauche
          Vect_D1 = [mean_DPas_D_f1, mean_LPas_D_f1, mean_Vitesse_D_f1, mean_SA_D_f1, mean_PA_D_f1, std_DPas_D_f1, std_LPas_D_f1, std_PA_D_f1, std_SA_D_f1, std_Vitesse_D_f1]
          Vect_G1 = [mean_DPas_G_f1, mean_LPas_G_f1, mean_Vitesse_G_f1, mean_SA_G_f1, mean_PA_G_f1, std_DPas_G_f1, std_LPas_G_f1, std_PA_G_f1, std_SA_G_f1, std_Vitesse_G_f1]
        
          # Réalisation des somprod et obtention de la projection du sujet
          SP_D1 = np.sum(Vect_D1 * cn, axis=0)
          SP_G1 = np.sum(Vect_G1 * cn, axis=0)
        
          Diff_Sujet_CTRL_G1 = (SP_G1-Projection_CTRL)
          Diff_Sujet_CTRL_G21 = abs(Diff_Sujet_CTRL_G1)+1
          Diff_Sujet_CTRL_D1 = (SP_D1-Projection_CTRL);
          Diff_Sujet_CTRL_D21 = abs(Diff_Sujet_CTRL_D1)+1
        
          # Fin et calcul eGVI
          if Diff_Sujet_CTRL_G1 < 0 :
              ln_sujet_G1 = -np.log(Diff_Sujet_CTRL_G21)
          else :
              ln_sujet_G1 = np.log(Diff_Sujet_CTRL_G21)
        
          if Diff_Sujet_CTRL_D1 < 0 :
              ln_sujet_D1 = -np.log(Diff_Sujet_CTRL_D21)
          else :
              ln_sujet_D1 = np.log(Diff_Sujet_CTRL_D21)
        
          # z gauche
          z_G1 = []  
          if ln_sujet_G1 < -m_ln_CTRL :
              z_G1 = (ln_sujet_G1 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_G1 > m_ln_CTRL :
              z_G1 = (ln_sujet_G1 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_G1 and ln_sujet_G1 < m_ln_CTRL :
              z_G1 = 0
        
          # z droit
          z_D1 = []  
          if ln_sujet_D1 < -m_ln_CTRL :
              z_D1 = (ln_sujet_D1 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_D1 > m_ln_CTRL :
              z_D1 = (ln_sujet_D1 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_D1 and  ln_sujet_D1 < m_ln_CTRL :
              z_D1 = 0
        
          eGVI_G1 = 100+z_G1
          eGVI_D1 = 100+z_D1
          eGVI1 = (eGVI_D1 + eGVI_G1)/2
        
        # COTE DROIT
        PA_D2 = []
        SA_D2 = []
        LPas_D2 = []
        Vitesse_D2 = []
        DPas_D2 = []
        
        for i,j in rhee_valid_cycles2 :
          PA_D2.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_D2.append(np.abs((data2[:,labels2.index('RTOE'),:][1, n] - data2[:,labels2.index('LTOE'),:][1, n])/10))
          Vitesse_D2.append(np.abs(((data2[:,labels2.index('RTOE'),:][1, n] - data2[:,labels2.index('RTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in lhee_valid_cycles2 :
            DPas_D2.append(np.abs(k - i)/100)
            SA_D2.append(np.abs(k-m))
        
        if len(DPas_D2) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_D2.pop(i)
            SA_D2.pop(i)
        
        elif len(DPas_D2) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_D2.pop(i)
            SA_D2.pop(i)
        
        elif len(DPas_D2) == 3:
          error_acq2 = True
        
          # COTE GAUCHE
        PA_G2 = []
        SA_G2 = []
        LPas_G2 = []
        Vitesse_G2 = []
        DPas_G2 = []
        
        for i,j in lhee_valid_cycles2 :
          PA_G2.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_G2.append(np.abs((data2[:,labels2.index('LTOE'),:][1, n] - data2[:,labels2.index('RTOE'),:][1, n])/10))
          Vitesse_G2.append(np.abs(((data2[:,labels2.index('LTOE'),:][1, n] - data2[:,labels2.index('LTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in rhee_valid_cycles2 :
            DPas_G2.append(np.abs(k - i)/100)
            SA_G2.append(np.abs(k-m))
        
        if len(DPas_G2) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_G2.pop(i)
            SA_G2.pop(i)
        
        elif len(DPas_G2) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_G2.pop(i)
            SA_G2.pop(i)
        
        elif len(DPas_G2) == 3:
          error_acq2 = True
        
        if error_acq2 == False:
          # Valeur moyenne
          PA_D_m2 = np.mean(PA_D2)
          LPas_D_m2 = np.mean(LPas_D2)
          DPas_D_m2 = np.mean(DPas_D2)
          Vitesse_D_m2 = np.mean(Vitesse_D2)
          SA_D_m2 = np.mean(SA_D2)
        
          # En pourcentage de la valeur moyenne
          pPA_Dm2 = []
          pDPas_Dm2 = []
          pLPas_Dm2 = []
          pVitesse_Dm2 = []
          pSA_Dm2 = []
          if len(PA_D2) < len(DPas_D2) : 
            for i in range(0,len(PA_D2),1) :
              pPA_Dm2.append(PA_D2[i] * 100 / PA_D_m2)
              pDPas_Dm2.append(DPas_D2[i] * 100 / DPas_D_m2)
              pLPas_Dm2.append(LPas_D2[i] * 100 / LPas_D_m2)
              pVitesse_Dm2.append(Vitesse_D2[i] * 100 / Vitesse_D_m2)
              pSA_Dm2.append(SA_D2[i] * 100 / SA_D_m2)
          else : 
            for i in range(0,len(DPas_D2),1) :
              pPA_Dm2.append(PA_D2[i] * 100 / PA_D_m2)
              pDPas_Dm2.append(DPas_D2[i] * 100 / DPas_D_m2)
              pLPas_Dm2.append(LPas_D2[i] * 100 / LPas_D_m2)
              pVitesse_Dm2.append(Vitesse_D2[i] * 100 / Vitesse_D_m2)
              pSA_Dm2.append(SA_D2[i] * 100 / SA_D_m2)
        
          # Différence absolue
          PA_D_f2 = []
          DPas_D_f2 = []
          LPas_D_f2 = []
          Vitesse_D_f2 = []
          SA_D_f2 = []
        
          if len(PA_D2) < len(DPas_D2) :
            for i in range(0,len(PA_D2)-1,1) :
              PA_D_f2.append(np.abs(pPA_Dm2[i+1] - pPA_Dm2[i]))
              DPas_D_f2.append(np.abs(pDPas_Dm2[i+1] - pDPas_Dm2[i]))
              LPas_D_f2.append(np.abs(pLPas_Dm2[i+1] - pLPas_Dm2[i]))
              Vitesse_D_f2.append(np.abs(pVitesse_Dm2[i+1] - pVitesse_Dm2[i]))
              SA_D_f2.append(np.abs(pSA_Dm2[i+1] - pSA_Dm2[i]))
          else : 
            for i in range(0,len(DPas_D2)-1,1) :
              PA_D_f2.append(np.abs(pPA_Dm2[i+1] - pPA_Dm2[i]))
              DPas_D_f2.append(np.abs(pDPas_Dm2[i+1] - pDPas_Dm2[i]))
              LPas_D_f2.append(np.abs(pLPas_Dm2[i+1] - pLPas_Dm2[i]))
              Vitesse_D_f2.append(np.abs(pVitesse_Dm2[i+1] - pVitesse_Dm2[i]))
              SA_D_f2.append(np.abs(pSA_Dm2[i+1] - pSA_Dm2[i]))
        
          # Valeur moyenne
          PA_G_m2 = np.mean(PA_G2)
          LPas_G_m2 = np.mean(LPas_G2)
          DPas_G_m2 = np.mean(DPas_G2)
          Vitesse_G_m2 = np.mean(Vitesse_G2)
          SA_G_m2 = np.mean(SA_G2)
        
          # En pourcentage de la valeur moyenne
          pPA_Gm2 = []
          pDPas_Gm2 = []
          pLPas_Gm2 = []
          pVitesse_Gm2 = []
          pSA_Gm2 = []
          if len(PA_G2) < len(DPas_G2) : 
            for i in range(0,len(PA_G2),1) :
              pPA_Gm2.append(PA_G2[i] * 100 / PA_G_m2)
              pDPas_Gm2.append(DPas_G2[i] * 100 / DPas_G_m2)
              pLPas_Gm2.append(LPas_G2[i] * 100 / LPas_G_m2)
              pVitesse_Gm2.append(Vitesse_G2[i] * 100 / Vitesse_G_m2)
              pSA_Gm2.append(SA_G2[i] * 100 / SA_G_m2)
          else : 
            for i in range(0,len(DPas_G2),1) :
              pPA_Gm2.append(PA_G2[i] * 100 / PA_G_m2)
              pDPas_Gm2.append(DPas_G2[i] * 100 / DPas_G_m2)
              pLPas_Gm2.append(LPas_G2[i] * 100 / LPas_G_m2)
              pVitesse_Gm2.append(Vitesse_G2[i] * 100 / Vitesse_G_m2)
              pSA_Gm2.append(SA_G2[i] * 100 / SA_G_m2)
        
          # Différence absolue
          PA_G_f2 = []
          DPas_G_f2 = []
          LPas_G_f2 = []
          Vitesse_G_f2 = []
          SA_G_f2 = []
        
          if len(PA_G2) < len(DPas_G2) :
            for i in range(0,len(PA_G2)-1,1) :
              PA_G_f2.append(np.abs(pPA_Gm2[i+1] - pPA_Gm2[i]))
              DPas_G_f2.append(np.abs(pDPas_Gm2[i+1] - pDPas_Gm2[i]))
              LPas_G_f2.append(np.abs(pLPas_Gm2[i+1] - pLPas_Gm2[i]))
              Vitesse_G_f2.append(np.abs(pVitesse_Gm2[i+1] - pVitesse_Gm2[i]))
              SA_G_f2.append(np.abs(pSA_Gm2[i+1] - pSA_Gm2[i]))
          else : 
            for i in range(0,len(DPas_G2)-1,1) :
              PA_G_f2.append(np.abs(pPA_Gm2[i+1] - pPA_Gm2[i]))
              DPas_G_f2.append(np.abs(pDPas_Gm2[i+1] - pDPas_Gm2[i]))
              LPas_G_f2.append(np.abs(pLPas_Gm2[i+1] - pLPas_Gm2[i]))
              Vitesse_G_f2.append(np.abs(pVitesse_Gm2[i+1] - pVitesse_Gm2[i]))
              SA_G_f2.append(np.abs(pSA_Gm2[i+1] - pSA_Gm2[i]))
        
          #Moyenne des différences absolues
          # Droit :
          mean_PA_D_f2 = np.mean(PA_D_f2)
          mean_DPas_D_f2 = np.mean(DPas_D_f2)
          mean_LPas_D_f2 = np.mean(LPas_D_f2)
          mean_Vitesse_D_f2 = np.mean(Vitesse_D_f2)
          mean_SA_D_f2 = np.mean(SA_D_f2)
        
          # Gauche :
          mean_PA_G_f2 = np.mean(PA_G_f2)
          mean_DPas_G_f2 = np.mean(DPas_G_f2)
          mean_LPas_G_f2 = np.mean(LPas_G_f2)
          mean_Vitesse_G_f2 = np.mean(Vitesse_G_f2)
          mean_SA_G_f2 = np.mean(SA_G_f2)
        
          # Ecart-type des différences absolues
          # Droit :
          std_PA_D_f2 = np.std(PA_D_f2)
          std_DPas_D_f2 = np.std(DPas_D_f2)
          std_LPas_D_f2 = np.std(LPas_D_f2)
          std_Vitesse_D_f2 = np.std(Vitesse_D_f2)
          std_SA_D_f2 = np.std(SA_D_f2)
        
          # Gauche :
          std_PA_G_f2 = np.std(PA_G_f2)
          std_DPas_G_f2 = np.std(DPas_G_f2)
          std_LPas_G_f2 = np.std(LPas_G_f2)
          std_Vitesse_G_f2 = np.std(Vitesse_G_f2)
          std_SA_G_f2 = np.std(SA_G_f2)
        
          # Création des vecteurs droit et gauche
          Vect_D2 = [mean_DPas_D_f2, mean_LPas_D_f2, mean_Vitesse_D_f2, mean_SA_D_f2, mean_PA_D_f2, std_DPas_D_f2, std_LPas_D_f2, std_PA_D_f2, std_SA_D_f2, std_Vitesse_D_f2]
          Vect_G2 = [mean_DPas_G_f2, mean_LPas_G_f2, mean_Vitesse_G_f2, mean_SA_G_f2, mean_PA_G_f2, std_DPas_G_f2, std_LPas_G_f2, std_PA_G_f2, std_SA_G_f2, std_Vitesse_G_f2]
        
          # Réalisation des somprod et obtention de la projection du sujet
          SP_D2 = np.sum(Vect_D2 * cn, axis=0)
          SP_G2 = np.sum(Vect_G2 * cn, axis=0)
        
          Diff_Sujet_CTRL_G2 = (SP_G2-Projection_CTRL)
          Diff_Sujet_CTRL_G22 = abs(Diff_Sujet_CTRL_G2)+1
          Diff_Sujet_CTRL_D2 = (SP_D2-Projection_CTRL);
          Diff_Sujet_CTRL_D22 = abs(Diff_Sujet_CTRL_D2)+1
        
          # Fin et calcul eGVI
          if Diff_Sujet_CTRL_G2 < 0 :
              ln_sujet_G2 = -np.log(Diff_Sujet_CTRL_G22)
          else :
              ln_sujet_G2 = np.log(Diff_Sujet_CTRL_G22)
        
          if Diff_Sujet_CTRL_D2 < 0 :
              ln_sujet_D2 = -np.log(Diff_Sujet_CTRL_D22)
          else :
              ln_sujet_D2 = np.log(Diff_Sujet_CTRL_D22)
        
          # z gauche
          z_G2 = []  
          if ln_sujet_G2 < -m_ln_CTRL :
              z_G2 = (ln_sujet_G2 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_G2 > m_ln_CTRL :
              z_G2 = (ln_sujet_G2 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_G2 and ln_sujet_G2 < m_ln_CTRL :
              z_G2 = 0
        
          # z droit
          z_D2 = []  
          if ln_sujet_D2 < -m_ln_CTRL :
              z_D2 = (ln_sujet_D2 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_D2 > m_ln_CTRL :
              z_D2 = (ln_sujet_D2 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_D2 and  ln_sujet_D2 < m_ln_CTRL :
              z_D2 = 0
        
          eGVI_G2 = 100+z_G2
          eGVI_D2 = 100+z_D2
          eGVI2 = (eGVI_D2 + eGVI_G2)/2
        
        # COTE DROIT
        PA_D3 = []
        SA_D3 = []
        LPas_D3 = []
        Vitesse_D3 = []
        DPas_D3 = []
        
        for i,j in rhee_valid_cycles3 :
          PA_D3.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_D3.append(np.abs((data3[:,labels3.index('RTOE'),:][1, n] - data3[:,labels3.index('LTOE'),:][1, n])/10))
          Vitesse_D3.append(np.abs(((data3[:,labels3.index('RTOE'),:][1, n] - data3[:,labels3.index('RTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in lhee_valid_cycles3 :
            DPas_D3.append(np.abs(k - i)/100)
            SA_D3.append(np.abs(k-m))
        
        if len(DPas_D3) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_D3.pop(i)
            SA_D3.pop(i)
        
        elif len(DPas_D3) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_D3.pop(i)
            SA_D3.pop(i)
        
        elif len(DPas_D3) == 3:
          error_acq3 = True
        
          # COTE GAUCHE
        PA_G3 = []
        SA_G3 = []
        LPas_G3 = []
        Vitesse_G3 = []
        DPas_G3 = []
        
        for i,j in lhee_valid_cycles3 :
          PA_G3.append((j-i)/100)
          n = int(i+((j-i)/2))
          LPas_G3.append(np.abs((data3[:,labels3.index('LTOE'),:][1, n] - data3[:,labels3.index('RTOE'),:][1, n])/10))
          Vitesse_G3.append(np.abs(((data3[:,labels3.index('LTOE'),:][1, n] - data3[:,labels3.index('LTOE'),:][1, i])/10)/(n-i))*100)
          for k,m in rhee_valid_cycles3 :
            DPas_G3.append(np.abs(k - i)/100)
            SA_G3.append(np.abs(k-m))
        
        if len(DPas_G3) == 9:
          correction = [1,1,2,2,3,3]
          for i in correction :
            DPas_G3.pop(i)
            SA_G3.pop(i)
        
        elif len(DPas_G3) == 6:
          correction = [1,1,2,2]
          for i in correction :
            DPas_G3.pop(i)
            SA_G3.pop(i)
        
        elif len(DPas_G3) == 3:
          error_acq3 = True
        
        if error_acq3 == False:
          # Valeur moyenne
          PA_D_m3 = np.mean(PA_D3)
          LPas_D_m3 = np.mean(LPas_D3)
          DPas_D_m3 = np.mean(DPas_D3)
          Vitesse_D_m3 = np.mean(Vitesse_D3)
          SA_D_m3 = np.mean(SA_D3)
        
          # En pourcentage de la valeur moyenne
          pPA_Dm3 = []
          pDPas_Dm3 = []
          pLPas_Dm3 = []
          pVitesse_Dm3 = []
          pSA_Dm3 = []
          if len(PA_D3) < len(DPas_D3) : 
            for i in range(0,len(PA_D3),1) :
              pPA_Dm3.append(PA_D3[i] * 100 / PA_D_m3)
              pDPas_Dm3.append(DPas_D3[i] * 100 / DPas_D_m3)
              pLPas_Dm3.append(LPas_D3[i] * 100 / LPas_D_m3)
              pVitesse_Dm3.append(Vitesse_D3[i] * 100 / Vitesse_D_m3)
              pSA_Dm3.append(SA_D3[i] * 100 / SA_D_m3)
          else : 
            for i in range(0,len(DPas_D3),1) :
              pPA_Dm3.append(PA_D3[i] * 100 / PA_D_m3)
              pDPas_Dm3.append(DPas_D3[i] * 100 / DPas_D_m3)
              pLPas_Dm3.append(LPas_D3[i] * 100 / LPas_D_m3)
              pVitesse_Dm3.append(Vitesse_D3[i] * 100 / Vitesse_D_m3)
              pSA_Dm3.append(SA_D3[i] * 100 / SA_D_m3)
        
          # Différence absolue
          PA_D_f3 = []
          DPas_D_f3 = []
          LPas_D_f3 = []
          Vitesse_D_f3 = []
          SA_D_f3 = []
        
          if len(PA_D3) < len(DPas_D3) :
            for i in range(0,len(PA_D3)-1,1) :
              PA_D_f3.append(np.abs(pPA_Dm3[i+1] - pPA_Dm3[i]))
              DPas_D_f3.append(np.abs(pDPas_Dm3[i+1] - pDPas_Dm3[i]))
              LPas_D_f3.append(np.abs(pLPas_Dm3[i+1] - pLPas_Dm3[i]))
              Vitesse_D_f3.append(np.abs(pVitesse_Dm3[i+1] - pVitesse_Dm3[i]))
              SA_D_f3.append(np.abs(pSA_Dm3[i+1] - pSA_Dm3[i]))
          else : 
            for i in range(0,len(DPas_D3)-1,1) :
              PA_D_f3.append(np.abs(pPA_Dm3[i+1] - pPA_Dm3[i]))
              DPas_D_f3.append(np.abs(pDPas_Dm3[i+1] - pDPas_Dm3[i]))
              LPas_D_f3.append(np.abs(pLPas_Dm3[i+1] - pLPas_Dm3[i]))
              Vitesse_D_f3.append(np.abs(pVitesse_Dm3[i+1] - pVitesse_Dm3[i]))
              SA_D_f3.append(np.abs(pSA_Dm3[i+1] - pSA_Dm3[i]))
        
          # Valeur moyenne
          PA_G_m3 = np.mean(PA_G3)
          LPas_G_m3 = np.mean(LPas_G3)
          DPas_G_m3 = np.mean(DPas_G3)
          Vitesse_G_m3 = np.mean(Vitesse_G3)
          SA_G_m3 = np.mean(SA_G3)
        
          # En pourcentage de la valeur moyenne
          pPA_Gm3 = []
          pDPas_Gm3 = []
          pLPas_Gm3 = []
          pVitesse_Gm3 = []
          pSA_Gm3 = []
          if len(PA_G3) < len(DPas_G3) : 
            for i in range(0,len(PA_G3),1) :
              pPA_Gm3.append(PA_G3[i] * 100 / PA_G_m3)
              pDPas_Gm3.append(DPas_G3[i] * 100 / DPas_G_m3)
              pLPas_Gm3.append(LPas_G3[i] * 100 / LPas_G_m3)
              pVitesse_Gm3.append(Vitesse_G3[i] * 100 / Vitesse_G_m3)
              pSA_Gm3.append(SA_G3[i] * 100 / SA_G_m3)
          else : 
            for i in range(0,len(DPas_G3),1) :
              pPA_Gm3.append(PA_G3[i] * 100 / PA_G_m3)
              pDPas_Gm3.append(DPas_G3[i] * 100 / DPas_G_m3)
              pLPas_Gm3.append(LPas_G3[i] * 100 / LPas_G_m3)
              pVitesse_Gm3.append(Vitesse_G3[i] * 100 / Vitesse_G_m3)
              pSA_Gm3.append(SA_G3[i] * 100 / SA_G_m3)
        
          # Différence absolue
          PA_G_f3 = []
          DPas_G_f3 = []
          LPas_G_f3 = []
          Vitesse_G_f3 = []
          SA_G_f3 = []
        
          if len(PA_G3) < len(DPas_G3) :
            for i in range(0,len(PA_G3)-1,1) :
              PA_G_f3.append(np.abs(pPA_Gm3[i+1] - pPA_Gm3[i]))
              DPas_G_f3.append(np.abs(pDPas_Gm3[i+1] - pDPas_Gm3[i]))
              LPas_G_f3.append(np.abs(pLPas_Gm3[i+1] - pLPas_Gm3[i]))
              Vitesse_G_f3.append(np.abs(pVitesse_Gm3[i+1] - pVitesse_Gm3[i]))
              SA_G_f3.append(np.abs(pSA_Gm3[i+1] - pSA_Gm3[i]))
          else : 
            for i in range(0,len(DPas_G3)-1,1) :
              PA_G_f3.append(np.abs(pPA_Gm3[i+1] - pPA_Gm3[i]))
              DPas_G_f3.append(np.abs(pDPas_Gm3[i+1] - pDPas_Gm3[i]))
              LPas_G_f3.append(np.abs(pLPas_Gm3[i+1] - pLPas_Gm3[i]))
              Vitesse_G_f3.append(np.abs(pVitesse_Gm3[i+1] - pVitesse_Gm3[i]))
              SA_G_f3.append(np.abs(pSA_Gm3[i+1] - pSA_Gm3[i]))
        
          #Moyenne des différences absolues
          # Droit :
          mean_PA_D_f3 = np.mean(PA_D_f3)
          mean_DPas_D_f3 = np.mean(DPas_D_f3)
          mean_LPas_D_f3 = np.mean(LPas_D_f3)
          mean_Vitesse_D_f3 = np.mean(Vitesse_D_f3)
          mean_SA_D_f3 = np.mean(SA_D_f3)
        
          # Gauche :
          mean_PA_G_f3 = np.mean(PA_G_f3)
          mean_DPas_G_f3 = np.mean(DPas_G_f3)
          mean_LPas_G_f3 = np.mean(LPas_G_f3)
          mean_Vitesse_G_f3 = np.mean(Vitesse_G_f3)
          mean_SA_G_f3 = np.mean(SA_G_f3)
        
          # Ecart-type des différences absolues
          # Droit :
          std_PA_D_f3 = np.std(PA_D_f3)
          std_DPas_D_f3 = np.std(DPas_D_f3)
          std_LPas_D_f3 = np.std(LPas_D_f3)
          std_Vitesse_D_f3 = np.std(Vitesse_D_f3)
          std_SA_D_f3 = np.std(SA_D_f3)
        
          # Gauche :
          std_PA_G_f3 = np.std(PA_G_f3)
          std_DPas_G_f3 = np.std(DPas_G_f3)
          std_LPas_G_f3 = np.std(LPas_G_f3)
          std_Vitesse_G_f3 = np.std(Vitesse_G_f3)
          std_SA_G_f3 = np.std(SA_G_f3)
        
          # Création des vecteurs droit et gauche
          Vect_D3 = [mean_DPas_D_f3, mean_LPas_D_f3, mean_Vitesse_D_f3, mean_SA_D_f3, mean_PA_D_f3, std_DPas_D_f3, std_LPas_D_f3, std_PA_D_f3, std_SA_D_f3, std_Vitesse_D_f3]
          Vect_G3 = [mean_DPas_G_f3, mean_LPas_G_f3, mean_Vitesse_G_f3, mean_SA_G_f3, mean_PA_G_f3, std_DPas_G_f3, std_LPas_G_f3, std_PA_G_f3, std_SA_G_f3, std_Vitesse_G_f3]
        
          # Réalisation des somprod et obtention de la projection du sujet
          SP_D3 = np.sum(Vect_D3 * cn, axis=0)
          SP_G3 = np.sum(Vect_G3 * cn, axis=0)
        
          Diff_Sujet_CTRL_G3 = (SP_G3-Projection_CTRL)
          Diff_Sujet_CTRL_G23 = abs(Diff_Sujet_CTRL_G3)+1
          Diff_Sujet_CTRL_D3 = (SP_D3-Projection_CTRL);
          Diff_Sujet_CTRL_D23 = abs(Diff_Sujet_CTRL_D3)+1
        
          # Fin et calcul eGVI
          if Diff_Sujet_CTRL_G3 < 0 :
              ln_sujet_G3 = -np.log(Diff_Sujet_CTRL_G23)
          else :
              ln_sujet_G3 = np.log(Diff_Sujet_CTRL_G23)
        
          if Diff_Sujet_CTRL_D3 < 0 :
              ln_sujet_D3 = -np.log(Diff_Sujet_CTRL_D23)
          else :
              ln_sujet_D3 = np.log(Diff_Sujet_CTRL_D23)
        
          # z gauche
          z_G3 = []
          if ln_sujet_G3 < -m_ln_CTRL :
              z_G3 = (ln_sujet_G3 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_G3 > m_ln_CTRL :
              z_G3 = (ln_sujet_G3 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_G3 and ln_sujet_G3 < m_ln_CTRL :
              z_G3 = 0
        
          # z droit
          z_D3 = []
          if ln_sujet_D3 < -m_ln_CTRL :
              z_D3 = (ln_sujet_D3 + m_ln_CTRL) / sd_ln_CTRL
        
          if ln_sujet_D3 > m_ln_CTRL :
              z_D3 = (ln_sujet_D3 - m_ln_CTRL) / sd_ln_CTRL
        
          if -m_ln_CTRL < ln_sujet_D3 and  ln_sujet_D3 < m_ln_CTRL :
              z_D3 = 0
        
          eGVI_G3 = 100+z_G3
          eGVI_D3 = 100+z_D3
          eGVI3 = (eGVI_D3 + eGVI_G3)/2
        
        Totegvi = []
        if error_acq1 == False :
          Totegvi.append(eGVI1)
        if error_acq2 == False :
          Totegvi.append(eGVI2)
        if error_acq3 == False :
          Totegvi.append(eGVI3)
        
        eGVI_M = round(np.mean(Totegvi),2)
        STD_eGVI_M = round(np.std(Totegvi),2)

        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"**Score eGVI Moyen** : {eGVI_M:.2f} +/- {STD_eGVI_M}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 98 et 102. Tout score en-dehors indique une atteinte à la variabilité de la marche.")

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")

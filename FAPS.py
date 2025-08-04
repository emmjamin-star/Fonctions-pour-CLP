import streamlit as st
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Score FAPS", layout="centered")
st.title("🦿 Score FAPS - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier de statique et un d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)
st.header("2. Indiquer le score allant de 0 (aucune aide à la marche) à 5 (participant totalement dépendant) pour les aides ambulatoire (case 1) et les dispositifs d'assistances (case 2)")
Score = [0, 1, 2, 3, 4, 5]
AmbulatoryAids = st.selectbox(
    "Pour l'aide ambulatoire :",
    Score['Score'])
    
AssistiveDevice = st.selectbox(
    "Pour le dispositif d'assistance :",
    Score['Score'])

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique = st.selectbox("Choisissez un fichier dynamique pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique.read())
        tmpd_path = tmp.name

    acq1 = ezc3d.c3d(tmpd_path)  # acquisition dynamique
    labels = acq1['parameters']['POINT']['LABELS']['value']
    freq = acq1['header']['points']['frame_rate']
    first_frame = acq1['header']['points']['first_frame']
    n_frames = acq1['data']['points'].shape[2]
    time_offset = first_frame / freq
    time = np.arange(n_frames) / freq + time_offset
    
    statique = ezc3d.c3d(tmp_path)  # acquisition statique
    labelsStat = statique['parameters']['POINT']['LABELS']['value']
    first_frameStat = statique['header']['points']['first_frame']
    n_framesStat = statique['data']['points'].shape[2]
    time_offsetStat = first_frameStat / freq
    timeStat = np.arange(n_framesStat) / freq + time_offsetStat
    
    markersStat  = statique['data']['points']
    markers1 = acq1['data']['points']

if st.button("Lancer le calcul du score FAPS"):
  try:
      # Extraction des coordonnées
      a1, a2, b1, b2, c1, c2 = markersStat[:,labels.index('LASI'),:][0, 0], markersStat[:,labels.index('LANK'),:][0, 0], markersStat[:,labels.index('LASI'),:][1, 0], markersStat[:,labels.index('LANK'),:][1, 0], markersStat[:,labels.index('LASI'),:][2, 0], markersStat[:,labels.index('LANK'),:][2, 0]
      LgJambeL = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
      
      a1, a2, b1, b2, c1, c2 = markersStat[:,labels.index('RASI'),:][0, 0], markersStat[:,labels.index('RANK'),:][0, 0], markersStat[:,labels.index('RASI'),:][1, 0], markersStat[:,labels.index('RANK'),:][1, 0], markersStat[:,labels.index('RASI'),:][2, 0], markersStat[:,labels.index('RANK'),:][2, 0]
      LgJambeR = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
  
      # Détection event gauche
      # Détection des cycles à partir du marqueur LHEE (talon gauche)
      points = acq1['data']['points']
      if "LHEE" in labels:
          idx_lhee = labels.index("LHEE")
          z_lhee = points[2, idx_lhee, :]
      
          # Inversion signal pour détecter les minima (contacts au sol)
          inverted_z = -z_lhee
          min_distance = int(freq * 0.8)
      
          # Détection pics
          peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
      
          # Début et fin des cycles = entre chaque pic
          lhee_cycle_start_indices = peaks[:-1]
          lhee_cycle_end_indices = peaks[1:]
          min_lhee_cycle_duration = int(0.5 * freq)
          lhee_valid_cycles = [
            (start, end) for start, end in zip(lhee_cycle_start_indices, lhee_cycle_end_indices)
            if (end - start) >= min_lhee_cycle_duration
          ]
          lhee_n_cycles = len(lhee_valid_cycles)
  
      # Détection event droite
      # Détection des cycles à partir du marqueur RHEE (talon droite)
      points = acq1['data']['points']
      if "RHEE" in labels:
          idx_rhee = labels.index("RHEE")
          z_rhee = points[2, idx_rhee, :]
      
          # Inversion signal pour détecter les minima (contacts au sol)
          inverted_z = -z_rhee
          min_distance = int(freq * 0.8)
      
          # Détection pics
          peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
      
          # Début et fin des cycles = entre chaque pic
          rhee_cycle_start_indices = peaks[:-1]
          rhee_cycle_end_indices = peaks[1:]
          min_rhee_cycle_duration = int(0.5 * freq)
          rhee_valid_cycles = [
            (start, end) for start, end in zip(rhee_cycle_start_indices, rhee_cycle_end_indices)
            if (end - start) >= min_rhee_cycle_duration
          ]
          rhee_n_cycles = len(rhee_valid_cycles)
  
      # Longueur pas à droite
      LgPasR = []
      for i,j in rhee_valid_cycles:
        a1, a2, b1, b2, c1, c2 = markers1[:,labels.index('RANK'),:][0,i], markers1[:,labels.index('RANK'),:][0,j], markers1[:,labels.index('RANK'),:][1,i], markers1[:,labels.index('RANK'),:][1,j], markers1[:,labels.index('RANK'),:][2,i], markers1[:,labels.index('RANK'),:][2,j]
        z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
        LgPasR.append(z)
      LgPasRmoy = np.mean(LgPasR)
      VarLgPr = np.std(LgPasR)
  
      #Longueur de pas à gauche
      LgPasG = []
      for i,j in lhee_valid_cycles:
        a1, a2, b1, b2, c1, c2 = markers1[:,labels.index('LANK'),:][0,i], markers1[:,labels.index('LANK'),:][0,j], markers1[:,labels.index('LANK'),:][1,i], markers1[:,labels.index('LANK'),:][1,j], markers1[:,labels.index('LANK'),:][2,i], markers1[:,labels.index('LANK'),:][2,j]
        z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
        LgPasG.append(z)
      LgPasLmoy = np.mean(LgPasG)
      VarLgPl = np.std(LgPasG)
  
      # Vitesse de marche
      Vmarche = ((markers1[:,labels.index('STRN'),:][0,-1]-markers1[:,labels.index('STRN'),:][0,0]) / (len(markers1[:,labels.index('STRN'),:][0,:]) / 100)) / 1000
  
      # Calcul du ratio et des indices liés à la marche
      RatioLL_SLr = (LgPasRmoy / 100) / 2
      RatioLL_SLl = (LgPasLmoy / 100) / 2
      RatioV_LLr = (Vmarche / (LgJambeR/1000))
      RatioV_LLl = (Vmarche / (LgJambeL/1000))
  
      # Temps de cycle
      StepTimeCycleR = []
      for  i,j in rhee_valid_cycles:
        z = (j - i)/freq
        StepTimeCycleR.append(z)
      
      StepTimeCycleL = []
      for  i,j in lhee_valid_cycles:
        z = (j - i)/freq
        StepTimeCycleL.append(z)
      
      StepTimer = np.mean(StepTimeCycleR)
      StepTimel = np.mean(StepTimeCycleL)
  
      # Base de soutien dynamique
      # Lors du pas  coté droit
      za =[]
      
      for i,j in rhee_valid_cycles :
        a2, b2, c2 = markers1[:,labels.index('LHEE'),:][0,i], markers1[:,labels.index('LHEE'),:][1,i], markers1[:,labels.index('LHEE'),:][2,i]
        a4, b4, c4 = markers1[:,labels.index('RHEE'),:][0,i], markers1[:,labels.index('RHEE'),:][1,i], markers1[:,labels.index('RHEE'),:][2,i]
        a3, b3, c3 = markers1[:,labels.index('RHEE'),:][0,j], markers1[:,labels.index('RHEE'),:][1,j], markers1[:,labels.index('RHEE'),:][2,j]
        a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
        z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
        za.append(z)
      
      DBSr = (np.mean(za))
      
      # Lors du pas  coté gauche
      za =[]
      
      for i,j in lhee_valid_cycles :
        a2, b2, c2 = markers1[:,labels.index('RHEE'),:][0,i], markers1[:,labels.index('RHEE'),:][1,i], markers1[:,labels.index('RHEE'),:][2,i]
        a4, b4, c4 = markers1[:,labels.index('LHEE'),:][0,i], markers1[:,labels.index('LHEE'),:][1,i], markers1[:,labels.index('LHEE'),:][2,i]
        a3, b3, c3 = markers1[:,labels.index('LHEE'),:][0,j], markers1[:,labels.index('LHEE'),:][1,j], markers1[:,labels.index('LHEE'),:][2,j]
        a1, b1, c1 =  (a3+a4)/2, (b3+b4)/2, (c3+c4)/2
        z = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
        za.append(z)
      
      DBSl = (np.mean(za))
      
      # Calcul sccore  Dynamique Base Support
      DynamiqueBaseSupport = ((DBSr+DBSl)/2)/2/100
      DynamiqueBaseSupport = np.abs(DynamiqueBaseSupport-10)
      if DynamiqueBaseSupport > 8:
        DynamiqueBaseSupport = 8
      DynamiqueBaseSupport = np.abs(DynamiqueBaseSupport - 8)
  
      # Step function L
      StepFunctionL = np.abs(RatioV_LLl - 1.49)/0.082 + np.abs(RatioLL_SLl - 0.77)/0.046 + np.abs(StepTimel - 0.52) / 0.028
      if StepFunctionL > 22 : 
        StepFunctionL = 22
      StepFunctionL = np.abs(StepFunctionL - 22)
  
      # Step function R
      StepFunctionR = np.abs(RatioV_LLr - 1.49)/0.082 + np.abs(RatioLL_SLr - 0.77)/0.046 + np.abs(StepTimer - 0.52) / 0.028
      if StepFunctionR > 22 : 
        StepFunctionR = 22
      StepFunctionR = np.abs(StepFunctionR - 22)
  
      # SL Asy
      SL_Asy = np.abs(RatioLL_SLr / RatioLL_SLl) / 0.2
      if SL_Asy > 8 : 
        SL_Asy = 8
      SL_Asy = np.abs(SL_Asy - 8)
  
      # Score final FAPS
      AssistiveDevice = int(input())
      AmbulatoryAids = int(input())
      ScoreFAPS = np.round(100 - (StepFunctionR + StepFunctionL + SL_Asy + DynamiqueBaseSupport + AmbulatoryAids + AssistiveDevice),2)

      st.markdown("### 📊 Résultats du score FAPS")
      st.write(f"**Score FAPS** : {ScoreFAPS:.3f}")
      st.write(f"**Fonction du pas gauche** : {StepFunctionL}")
      st.write(f"**Fonction du pas droit** : {StepFunctionR:.2f}")
      st.write(f"**Base de support dynamique** : {DynamiqueBaseSupport:.3f}")
      st.write(f"**Asymétrie** : {SL_Asy:.3f}")
      st.write(f"**Aide ambulatoire** : {AmbulatoryAids:.3f}")
      st.write(f"**Dispositif d'assistance** : {AssistiveDevice:.3f}")
      
  except Exception as e:
      st.error(f"Erreur pendant l'analyse : {e}")

  

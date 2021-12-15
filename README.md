# BioRadar: Contact-Free Vital Signs Monitoring using the XM112 Pulsed-Coherent Millimetre-Wave Radar Module

Link to statistical anlaysis of heart rate measurements in Python: https://colab.research.google.com/drive/1NaPGNI30N7ld4ZmqZBicBSW60nfZAZyp?usp=sharing

BioRadar is an IoMT device which measures respiration rate (RR), heart rate (HR) and heart rate variability (HRV) without any contact with the body. It uses Acconeer's XM112 mmWave radar module to precisely capture the movements of the chest, which is composed of 
* a large low-frequency respiratory signal from the inhalation and exhalation of the lungs, and 
* a tiny high-frequency heartbeat signal from the contraction and relaxation of the heart.

The XM112 is operated the IQ mode to allow for accurate phase estimation of the returning radio pulses. 

BioRadar offers two modes:
*  RR/HR monitoring (optimized for stable RR/HR measurements) 
*  High-res BCG (for accurate HRV measurement, requires subject to hold their breath)

The XM122 is enclosed in a small 3D-printed chassis.

Unlike past projects that have used Acconeer's radar sensors to monitor vital signs, BioRadar offers an unprecedented level of accuracy for simultaneously measuring HR and RR without requiring the subject to hold their breath. An FIR bandpass filter was used with a Kaiser window to extract the heartbeat signal from the large respiratory motions, followed by HR estimation by counting the number of peaks in the heartbeat in a 20 second window.

## Equipment
*  XM112/XB112 radar/breakout
*  LH122 lens kit (hyperbolic lens)
*  Raspberry Pi 4B 2GB

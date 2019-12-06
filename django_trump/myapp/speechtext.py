#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:05:34 2019

@author: sumidakouki
"""

import speech_recognition as sr
r = sr.Recognizer()
mic = sr.Microphone()

def SpeechToText():
   while True:
    print("Say something ...")

    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    print ("Now to recognize it...")

    try:
        print(r.recognize_google(audio, language='ja-JP'))
        
        if r.recognize_google(audio, language='ja-JP') in "ターンエンド" :
            print("end")
            break
    except sr.UnknownValueError:
        print("could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        
SpeechToText()
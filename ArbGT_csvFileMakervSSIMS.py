# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 08:03:46 2018

@author: msmanski

This script will generate CSV files from a compressed .pkl dataframe for All Mosquitoes,
all hatched mosquitoes (no eggs or gestating agents), or all adult mosquitoes. Line30 can be modified 
to report only adult females, provided the agent.sex information was recorded in the .pkl file

Genotypes reported include total, wild-type, SSIMS/FAMSS(same genotype), SGI (SS), FL, WT-FL hybrid, embyonic lethal,
promoter conversion resistant mutants (SSIMSRes), SGI-FL hybrids, and 'other genotypes'

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import winsound



def count_genotypes(genotypeList,StateGenPosData, x, y):
    """Counts how many of each genotype are present at each step"""
    allMos = 0
    nonEggs = 0
    Adults = 0
    for i in range(len(genotypeList)):
        gt = genotypeList[i]
        b = sum(1 for item in StateGenPosData if not 'new' in item[0] and not 'gestating' in item[0] and gt in item[1] and item[2]==(x,y))
        c = sum(1 for item in StateGenPosData if 'adult' in item[0]  and 'XX' in item[1] and not 'gestating' in item[0] and gt in item[1] and item[2]==(x,y))
        d = sum(1 for item in StateGenPosData if 'adult' in item[0] and gt in item[1] and item[2]==(x,y))
##        for item in StateGenPosData:
##            print(item[0],item[1],item[2])
##            if 'adult' in item[0] and gt in item[1] and item[2]==(x,y):
##                d+=1
##                print('yay')
##            if not 'new' in item[0] and not 'egg' in item[0] and not 'gestating' in item[0] and gt in item[1] and item[2]==(x,y):
##                c+=1
##            if not 'new' in item[0] and not 'gestating' in item[0] and gt in item[1] and item[2]==(x,y):
##                b+=1
        allMos = allMos + b
        nonEggs = nonEggs + c
        Adults = Adults + d
    return allMos, nonEggs, Adults


def processModelData(StateGenPosData, filename, cells, height):
    Xaxis = np.zeros((1,1))
    YaxisTotal_allMos = np.zeros((1,cells))
    YaxisWT_allMos = np.zeros((1,cells))
    YaxisSSIMS_allMos = np.zeros((1,cells))
    YaxisSS_FLhet_allMos = np.zeros((1,cells))
    YaxisFL_allMos = np.zeros((1,cells))
    #YaxisOtherGenotype_allMos = np.zeros((1,cells))
    YaxisTotal_nonEggs = np.zeros((1,cells))
    YaxisWT_nonEggs = np.zeros((1,cells))
    YaxisSSIMS_nonEggs = np.zeros((1,cells))
    YaxisSS_FLhet_nonEggs = np.zeros((1,cells))
    YaxisFL_nonEggs = np.zeros((1,cells))
    #YaxisOtherGenotype_nonEggs = np.zeros((1,cells))    
    YaxisTotal_Adults = np.zeros((1,cells))
    YaxisWT_Adults = np.zeros((1,cells))
    YaxisSSIMS_Adults = np.zeros((1,cells))
    YaxisSS_FLhet_Adults = np.zeros((1,cells))
    YaxisFL_Adults = np.zeros((1,cells))
    #YaxisOtherGenotype_Adults = np.zeros((1,cells))
    for i in range(len(StateGenPosData.axes[0].levels[0])):
        print(i/len(StateGenPosData.axes[0].levels[0])*100)
        Xaxis = np.vstack((Xaxis,i))
        totalRow_allMos = np.zeros((1,cells))
        WTRow_allMos = np.zeros((1,cells))
        SSIMSRow_allMos = np.zeros((1,cells))
        SS_FLhetRow_allMos = np.zeros((1,cells))
        FLRow_allMos = np.zeros((1,cells))
        #OtherGenotypeRow_allMos = np.zeros((1,cells))
        totalRow_nonEggs = np.zeros((1,cells))
        WTRow_nonEggs = np.zeros((1,cells))
        SSIMSRow_nonEggs = np.zeros((1,cells))
        SS_FLhetRow_nonEggs = np.zeros((1,cells))
        FLRow_nonEggs = np.zeros((1,cells))
        #OtherGenotypeRow_nonEggs = np.zeros((1,cells))
        totalRow_Adults = np.zeros((1,cells))
        WTRow_Adults = np.zeros((1,cells))
        SSIMSRow_Adults = np.zeros((1,cells))
        SS_FLhetRow_Adults = np.zeros((1,cells))
        FLRow_Adults = np.zeros((1,cells))
        #OtherGenotypeRow_Adults = np.zeros((1,cells))
        for a in range(height):
            for b in range(np.floor_divide(cells,height)):
                col = (height*a)+b
                try:
                    totalRow_allMos[0,col], totalRow_nonEggs[0,col], totalRow_Adults[0,col] = count_genotypes([''],StateGenPosData.loc[i].values, a, b) #total
                    WTRow_allMos[0,col], WTRow_nonEggs[0,col], WTRow_Adults[0,col] = count_genotypes(['bbddppttllWWXX','bbddppttllWWXY','bbddppttllWWYX'],StateGenPosData.loc[i].values, a, b)#Wt
                    SSIMSRow_allMos[0,col], SSIMSRow_nonEggs[0,col], SSIMSRow_Adults[0,col] = count_genotypes(['PPTTLL'],StateGenPosData.loc[i].values, a, b)#R Res Homo
                    SS_FLhetRow_allMos[0,col], SS_FLhetRow_nonEggs[0,col], SS_FLhetRow_Adults[0,col] = count_genotypes(['PPTTLl','PPTTlL'],StateGenPosData.loc[i].values, a, b)
                    FLRow_allMos[0,col], FLRow_nonEggs[0,col], FLRow_Adults[0,col] = count_genotypes(['ppttLl','ppttlL','ppttLL'],StateGenPosData.loc[i].values, a, b)
                    #OtherGenotypeRow_allMos[0,col], OtherGenotypeRow_nonEggs[0,col], OtherGenotypeRow_Adults[0,col] = count_genotypes(['PPtTll','pptTll','PPTtll','ppTtll','PPtTLl','pptTLl','PPTtLl','ppTtLl','PPtTlL','pptTlL','PPTtlL','ppTtlL','PPtTLL','pptTLL','PPTtLL','ppTtLL','pPTTll','PpTTll','pPttll','Ppttll','pPTTLl','PpTTLl','pPttLl','PpttLl','pPTTlL','PpTTlL','pPttlL','PpttlL','pPTTLL','PpTTLL','pPttLL','PpttLL','PPttll','ppTTll','PPttLl','ppTTLl','PPttlL','ppTTlL','PPttLL','ppTTLL'],StateGenPosData.loc[i].values, a, b)
                except:
                    print('AHHH')


        YaxisTotal_allMos = np.vstack((YaxisTotal_allMos,totalRow_allMos))
        YaxisWT_allMos = np.vstack((YaxisWT_allMos,WTRow_allMos))
        YaxisSSIMS_allMos = np.vstack((YaxisSSIMS_allMos,SSIMSRow_allMos))
        YaxisSS_FLhet_allMos = np.vstack((YaxisSS_FLhet_allMos,SS_FLhetRow_allMos))
        YaxisFL_allMos = np.vstack((YaxisFL_allMos,FLRow_allMos))
        #YaxisOtherGenotype_allMos = np.vstack((YaxisOtherGenotype_allMos,OtherGenotypeRow_allMos))
        
        YaxisTotal_nonEggs = np.vstack((YaxisTotal_nonEggs,totalRow_nonEggs))
        YaxisWT_nonEggs = np.vstack((YaxisWT_nonEggs,WTRow_nonEggs))
        YaxisSSIMS_nonEggs = np.vstack((YaxisSSIMS_nonEggs,SSIMSRow_nonEggs))
        YaxisSS_FLhet_nonEggs = np.vstack((YaxisSS_FLhet_nonEggs,SS_FLhetRow_nonEggs))
        YaxisFL_nonEggs = np.vstack((YaxisFL_nonEggs,FLRow_nonEggs))
        #YaxisOtherGenotype_nonEggs = np.vstack((YaxisOtherGenotype_nonEggs,OtherGenotypeRow_nonEggs))
        
        YaxisTotal_Adults = np.vstack((YaxisTotal_Adults,totalRow_Adults))
        YaxisWT_Adults = np.vstack((YaxisWT_Adults,WTRow_Adults))
        YaxisSSIMS_Adults = np.vstack((YaxisSSIMS_Adults,SSIMSRow_Adults))
        YaxisSS_FLhet_Adults = np.vstack((YaxisSS_FLhet_Adults,SS_FLhetRow_Adults))
        YaxisFL_Adults = np.vstack((YaxisFL_Adults,FLRow_Adults))
        #YaxisOtherGenotype_Adults = np.vstack((YaxisOtherGenotype_Adults,OtherGenotypeRow_Adults))

    filenameb = filename.split('.')[0] +'_allMos_Total.csv'
    np.savetxt(filenameb,YaxisTotal_allMos, delimiter=',')
    filenameb = filename.split('.')[0] +'_allMos_WT.csv'
    np.savetxt(filenameb,YaxisWT_allMos, delimiter=',')
    filenameb = filename.split('.')[0] +'_allMos_SSIMS.csv'
    np.savetxt(filenameb,YaxisSSIMS_allMos, delimiter=',')
    filenameb = filename.split('.')[0] +'_allMos_SS_FLhet.csv'
    np.savetxt(filenameb,YaxisSS_FLhet_allMos, delimiter=',')
    filenameb = filename.split('.')[0] +'_allMos_FL.csv'
    np.savetxt(filenameb,YaxisFL_allMos, delimiter=',')
    #filenameb = filename.split('.')[0] +'_allMos_OtherGenotype.csv'
    #np.savetxt(filenameb,YaxisOtherGenotype_allMos, delimiter=',')

    filenameb = filename.split('.')[0] +'_female_Total.csv'
    np.savetxt(filenameb,YaxisTotal_nonEggs, delimiter=',')
    filenameb = filename.split('.')[0] +'_female_WT.csv'
    np.savetxt(filenameb,YaxisWT_nonEggs, delimiter=',')
    filenameb = filename.split('.')[0] +'_female_SSIMS.csv'
    np.savetxt(filenameb,YaxisSSIMS_nonEggs, delimiter=',')
    filenameb = filename.split('.')[0] +'_female_SS_FLhet.csv'
    np.savetxt(filenameb,YaxisSS_FLhet_nonEggs, delimiter=',')
    filenameb = filename.split('.')[0] +'_female_FL.csv'
    np.savetxt(filenameb,YaxisFL_nonEggs, delimiter=',')
    #filenameb = filename.split('.')[0] +'_non_eggs_OtherGenotype.csv'
    #np.savetxt(filenameb,YaxisOtherGenotype_nonEggs, delimiter=',')

    filenameb = filename.split('.')[0] +'_Adults_Total.csv'
    np.savetxt(filenameb,YaxisTotal_Adults, delimiter=',')
    filenameb = filename.split('.')[0] +'_Adults_WT.csv'
    np.savetxt(filenameb,YaxisWT_Adults, delimiter=',')
    filenameb = filename.split('.')[0] +'_Adults_SSIMS.csv'
    np.savetxt(filenameb,YaxisSSIMS_Adults, delimiter=',')
    filenameb = filename.split('.')[0] +'_Adults_SS_FLhet.csv'
    np.savetxt(filenameb,YaxisSS_FLhet_Adults, delimiter=',')
    filenameb = filename.split('.')[0] +'_Adults_FL.csv'
    np.savetxt(filenameb,YaxisFL_Adults, delimiter=',')
    #filenameb = filename.split('.')[0] +'_Adults_OtherGenotype.csv'
    #np.savetxt(filenameb,YaxisOtherGenotype_Adults, delimiter=',')

files= glob.glob("*.pkl")
for q in files:
    filename = q
    print(filename)
    StateGenPosData = pd.read_pickle(filename)
    cells = 1                                                              # Number of populations being simulated
    height = 1        
    processModelData(StateGenPosData, filename, cells, height)

winsound.Beep(1000,200)
winsound.Beep(1000,200)
winsound.Beep(1000,200)

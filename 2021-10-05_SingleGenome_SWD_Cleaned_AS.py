# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 13:03:27 2018

@author: smanski

This code is set up to simulate a 3 x 1 grid of individual population models. Because
there is no migration between the populations, each can be treated as an independent experiment.

TODO: make it possible for gestating eggs to move with migration rate

Modified on Wed Mar 25 11:31:00 2020

@author: Adam Sychla

This modification allows running at any number of nhej probabilities.
    Comment was set by definition of nhej variable.
    The for loop of nhejVar was set to run at length of variable nhej

Modifications made in commenting to clarify steps (and make sections easier to find)

Modified on Fri Mar 27 12:31:00 2020

@author: Adam Sychla

Genome for SGD consolidated into single genome assumes independent assortment
Adds Sex Chromosomes

Modified on Wed Apr 08 22:35:00

@author: Adam Sychla

Single Genome expanded to allow programing of linkage dysequillibrium through LM matrix
Uses typeExp value to determine SSIMS vs. FAMSS

Modified on Thurs Jul 22 09:01:00

@author: Adam Sychla

Cap development between 10 and 30 C 30 set based on https://doi.org/10.1603/EN13200

Modified on Thurs Jul 22 09:15:00

@author: Adam Sychla

Added accidental mortality parameter as a function of temperature from https://link.springer.com/article/10.1007/s10340-015-0681-z

Modified on Mon Jul 26 13:21:00

@author: Adam Sychla

Adjusted development to that of https://link.springer.com/article/10.1007/s10340-015-0681-z

Temperature dependent fecundity from http://dx.doi.org/10.1093/jee/tow006

Modified on Tues Aug 17 12:49:00

@author: Adam Sychla

Switched to include second EGI construct

Modified on Tues Aug 17 13:10:00

@author: Adam Sychla

Multiple Matings possible (equal choice probability)

Modified on Tues Oct 05 08:26:30

@author: Adam Sychla

Fully allow multiple matings


"""
windows=True #if True imports winsound and alerts end of simulation. Set to false if NOT running on Windows

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import time
import math
if windows==True:
    import winsound
import sys



'''__________________________________________________
First the script will import a steady-state wild-type
population that will be used to seed each population
____________________________________________________'''


#seed = "seedDataNew.pkl"                                               
#StateGenPosData = pd.read_pickle(seed)
#seedData = StateGenPosData.loc[0].values

tempC = "tempData2010.csv"
TempData = pd.read_csv(tempC)
TempDataV2 = TempData = list(TempData['Temp'])
stepsize=1 #day equivalent of each timestep

'''__________________________________________________
Adjust the following parameters for a simulation of
interest.
__________________________________________________'''


#print(len(seedData))

#Locus Linkage Matrix range[0.5-1]=independent to fully linked/ Diagnol definitionally=1
LM = []
LM.append([1,0.5,0.5,0.5,0.5,0.5,0.5])#B locus
LM.append([0.5,1,0.5,0.5,0.5,0.5,0.5])#D locus
LM.append([0.5,0.5,1,0.5,0.5,0.5,0.5])#P locus
LM.append([0.5,0.5,0.5,1,0.5,0.5,0.5])#T locus
LM.append([0.5,0.5,0.5,0.5,1,0.5,0.5])#L locus
LM.append([0.5,0.5,0.5,0.5,0.5,1,0.5])#G locus
LM.append([0.5,0.5,0.5,0.5,0.5,0.5,1])#X chromosome

for nhejVar in range(1):


#Genotypes: p=wild-type promoter, P=engineered promoter G=Gene Drive R=Resistance Allele I=Resistant KO W=wild-type L=female lethal l=wild-type T=PTA t=wild-type U=DUpd+PTA u=Delta_Upd q=p promoter conversion r=natural mutation in p x=natural mutation in X
#Starting Genotypes will always start female.
#Upon generation of organsim last X may be switched to Y
#Allows for X linked GE
    startingGMOgenotype =  'bbddppttLLWWXX'     # SGD= 'bbddppttllGWXX', SSIMS = 'bbddPPTTLLWWXX', EGI = 'bbddPPTTllWWXX', FL = 'bbddppttLLWWXX', FAFL = bbdppttllWWUU
    wildtypeGenotype = 'ttllWWXX'         # for most simulations, wt = 'bbddppttllWWXX' NOTE:'bbddpp' is added further in the code to allow resistance


    #Resistance Frequencies
    HomingFreq = 0.98                   # Homing frequency (SGD), range: [0-1]
    NHEJfreq = 0            # NHEJ frequency (during gametogenesis, SGD), range: [0-1]
    promoterConversionFreq = 0          # Promoter Conversion Frequency (SSIMS/SGI), range: [0-1]
    mutationRate = 0                   # Mutation rate in targeted promoter

    #EDIT#
    Freq1 = 0                       #Rate of SNP1 in population
    Freq2 = 0                       #Rate of SNP2 in population

    startingGMO = 0                 #0 to account for reactionary release
    addedGMO = int(sys.argv[2])                        # intermittently added GE agents
    timeStepAdded = int(sys.argv[1])                 # number of time steps between intermittent addition.
    releaseCount = 0
    timeSteps = len(TempData)                     # number of time steps to simulate
    migrationRate = 0                   # range [0-1]; movement per adult per timestep;
    
    #Secondary Model parameters
    startingWT = 100          # wild-type agents placed at time step 0
    sexRatio = 0.50                     # range [0-1]; 0.25 = 25% females; 0.75 = 75% females
    GMsex = 0.5                           # range [0-1]; 0.25 = 25% females; 0.75 = 75% females
    
    gridWidth = 1                      # number of columns of population matrix
    gridHeight = 1                      # number of rows of population matrix
    borderCells = 0                     # width of 'Wild-type only' border
    
    #Density independent survival rates
    DayEggsPerFemale = 60
    MaxEggsPerFemale = 200

    carry=False # how much tet to raise the flies on. True==100mg/mL False==10mg/mL

    typeExp = 'FL'+str(startingGMO)+'start'+str(timeStepAdded)+'Release'+str(addedGMO)              # This is used for Filename generation '[typeExp]experiment[#].pkl'    
    
    '''____________________________________________
    End of Parameter Section; be careful if modifying
    values below
    ________________________________________________'''
    
    
    class Timer(object):
        def __init__(self, name=None):
            self.name = name
    
        def __enter__(self):
            self.tstart = time.time()
    
        def __exit__(self, type, value, traceback):
            if self.name:
                print('[{}]'.format(self.name))
            print('Elapsed: {}'.format(time.time() - self.tstart))
    
    class SWD(Agent):
        """A Spotted Wing Drosophila"""
        def __init__(self, unique_id, model):
            super().__init__(unique_id, model)
            self.lifeStage = 0                          
            self.state = 'egglv'
            self.pregnant = False
            self.mated = False
            self.gestating = 0
            self.mateGenotype = []
            self.moved = False
            self.fertile = True
            self.tet=False
            self.age = 0
            self.eggsLeft = MaxEggsPerFemale
            self.model.zzz += 1
    
            
        def mature(self):                                   # Progression from one life=stage to the next at each time step
            
            self.age += stepsize
            dayTemp=TempData[0]             #TempData has first data point removed with every timestep


            #Tempeture based mortality from Asplen et al (2015)
            Mfunc=stepsize*0.00035*(dayTemp-15)**2+0.01
            if random.random()<Mfunc and self.state!='dead':

                self.state = 'dead'
                self.lifeStage = 10000
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)
            
            #Max age: set at 100 days >> likely to survive. Left as a "clean up fucntion" in rare circumstance individual persists.
            if self.age >= 100 and self.state!='dead':

                self.state = 'dead'
                self.lifeStage = 10000
                self.model.grid._remove_agent(self.pos, self)
                self.model.schedule.remove(self)


            #progression of Egg/Larval Stage
            if self.state == 'egglv':


                if 'p' in self.genotype and 'T' in self.genotype:   #kill if wild-type promoter present with PTA
                    self.state = 'dead'
                    self.lifeStage = 10000
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)

                elif 'b' in self.genotype and 'D' in self.genotype:   #kill if wild-type promoter present with PTA
                    self.state = 'dead'
                    self.lifeStage = 10000
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)

                elif 'GG' in self.genotype:                         #kill if homozygous for Gene drive
                    self.lifeStage = 10000
                    self.state = 'dead'
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)
                    
                elif 'G' in self.genotype and 'I' in self.genotype: #Kill if no expression of WT gene for SGD
                    self.lifeStage = 10000
                    self.state = 'dead'
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)
                    
                elif 'U' in self.genotype and 'X' in self.genotype: #kill if FAFL allele present with Wt X
                    self.lifeStage = 10000
                    self.state = 'dead'
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)

                #development calculated from Asplen et al (2015) NOTE: 243 factor converts to degree-days
                dev = stepsize*234*(0.0044*(dayTemp-5.975))/(1+4.5**(dayTemp-31)) 
                self.lifeStage += dev
                y=self.lifeStage

                #Use Normal CDF centered around mean Egg degree days+ mean Larval degree days
                if random.random()<1/2*(1+math.erf((y-140.785)/(20.49*math.sqrt(2)))) and self.state!='dead':

                    self.state = 'pupa'
                    self.lifeStage = 0

            elif self.state == 'pupa':

                #development calculated from Asplen et al (2015) NOTE: 243 factor converts to degree-days
                dev = stepsize*234*(0.0044*(dayTemp-5.975))/(1+4.5**(dayTemp-31))
                self.lifeStage += dev
                y=self.lifeStage

                #Use Normal CDF centered around mean Pupal degree days
                if random.random()<1/2*(1+math.erf((y-93.22)/(6.18*math.sqrt(2)))):

                    self.state = 'adult'
                    self.lifeStage = 0


            elif self.state == 'adult':

                if 'L' in self.genotype and self.tet==False and self.sex == 'female': #Comment out '''and self.sex == 'female' ''' for RIDL           #kill by female lethality
                    self.lifeStage = 10000
                    self.state = 'dead'
                    self.model.grid._remove_agent(self.pos, self)
                    self.model.schedule.remove(self)

                else:

                    #development calculated from Asplen et al (2015) NOTE: 243 factor converts to degree-days
                    dev = stepsize*234*(0.0044*(dayTemp-5.975))/(1+4.5**(dayTemp-31))
                    self.lifeStage += dev
                    y=self.lifeStage

                    #Use Normal CDF centered around mean Adult degree days
                    if random.random()<1/2*(1+math.erf((y-1050)/(40.41*math.sqrt(2)))):

                        self.state = 'dead'
                        self.lifeStage = 10000
                        self.model.grid._remove_agent(self.pos, self)
                        self.model.schedule.remove(self)




            
            
                
        # Moves SWD between neighboring populations if simulating migration
        def move(self):                                                     
            possible_steps = self.model.grid.get_neighborhood(
                self.pos,
                moore=True,                                                 # includes diagonol moves, otherwise von neumann
                include_center=False)
            new_position = random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
            self.pos = list(new_position)
            self.moved = True

        # Produce new eggs with genotypes determined by parents    
        def reproduce(self, maleGenotype, female):  
            if female.state == 'adult':

                dayTemp=TempData[0]
                if self.eggsLeft>0 and dayTemp>5 and dayTemp<30:
                    

                    #number of eggs laid determined from Ryan et al (2016)
                    Eggs2Lay = stepsize*3.38313e-304*(2700.66-(dayTemp-22.87)**2)**88.53 #Simplified form of: 659.06*((88.53+1)/(3.14*52.32**(2*88.53+2))*(52.32**2-(dayTemp-22.87)**2-6.06**2)**88.53)

                    if Eggs2Lay > self.eggsLeft:

                        Eggs2Lay = self.eggsLeft

                    if random.random()<Eggs2Lay-math.floor(Eggs2Lay):
                        Eggs2Lay=math.floor(Eggs2Lay)+1

                    else:
                        Eggs2Lay=math.floor(Eggs2Lay)

                    self.eggsLeft-=Eggs2Lay

                else:

                    Eggs2Lay=0

                    
                for i in range(Eggs2Lay):
                    egg = SWD(self.model.zzz,self.model)
                    #generate Paternal Alleles to pass on
                    genSelect=random.randint(0,len(maleGenotype)-1)
                    maleGenotype2 = maleGenotype[genSelect]
                    
                    #Gene Drive Conversion Rules
                    if 'G' in maleGenotype2:                                     # Decide on inherited genotype from drive allele
                        if 'W' in maleGenotype2:
                            randNum = random.random()
                            
                            if randNum < np.multiply(HomingFreq,np.subtract(1,NHEJfreq)):
                                maleGenotype2=maleGenotype2.replace('W','G')
                                
                            elif np.multiply(HomingFreq,np.subtract(1,NHEJfreq)) < randNum < HomingFreq:
                                if random.random() < 0.33:                  
                                    maleGenotype2=maleGenotype2.replace('W','R')
                                    
                                else:
                                    maleGenotype2=maleGenotype2.replace('W','I')
                                    
                            else:
                                pass
                            
                        elif 'R' in maleGenotype2:                               #pass of R or G depending by linkage
                            pass
                        
                        else:
                            raise NameError('invalid Male Genotype')
                    #/Gene Drive Conversion Rules
                    
                    haploM=''

                    #Uses Linkage Matrix to generate Paternal Alleles
                    for x in range(int(len(maleGenotype2)/2)): #ensures haploid germcell is half length of diploid
                        P=0.5 #first allele is passed along with 50% rate
                        z=0
                        for y in haploM:

                            #determines if next allele to generate is linked to any of the previously generated alleles. Sets to closest linkage
                            P=0.5
                            z=maleGenotype2.index(y)/2
                            if LM[x][int(z)]>P:
                                P=LM[x][int(z)]

                        #determines which of the two alleles is passed along
                        if random.random()<P:
                            if int(z)==z:
                                haploM+=maleGenotype2[x*2]
                            else:
                                haploM+=maleGenotype2[x*2+1]

                        else:
                            if int(z)==z:
                                haploM+=maleGenotype2[x*2+1]
                            else:
                                haploM+=maleGenotype2[x*2]
                                
                    #/generate Paternal Alleles to pass on
                                
                    #generate Maternal Allele to pass on
                    femaleGenotype = female.genotype
                    
                    #Gene Drive Conversion Rules
                    if 'G' in femaleGenotype:                                     # Decide on inherited genotype from drive allele
                        if 'W' in femaleGenotype:
                            randNum = random.random()
                            if randNum < np.multiply(HomingFreq,np.subtract(1,NHEJfreq)):
                                femaleGenotype=femaleGenotype.replace('W','G')
                            elif np.multiply(HomingFreq,np.subtract(1,NHEJfreq)) < randNum < HomingFreq:
                                if random.random() < 0.33:                  
                                    femaleGenotype=femaleGenotype.replace('W','R')
                                else:
                                    femaleGenotype=femaleGenotype.replace('W','I')
                            else:
                                pass
                            
                        elif 'R' in femaleGenotype:                               #pass of R or G depending by Mendelian genetics
                            pass
                        
                        else:
                            raise NameError('invalid feMale Genotype')
                    #/Gene Drive Conversion Rules

                    haploF=''

                    #Uses Linkage Matrix to generate Maternal Alleles
                    for x in range(int(len(femaleGenotype)/2)):
                        P=0.5 #first allele is passed along with 50% rate
                        z=0
                        
                        for y in haploF:

                            #determines if next allele to generate is linked to any of the previously generated alleles. Sets to closest linkage
                            P=0.5
                            z=femaleGenotype.index(y)/2
                            if LM[x][int(z)]>P:
                                P=LM[x][int(z)]

                        #determines which of the two alleles is passed along
                        if random.random()<P:
                            if int(z)==z:
                                haploF+=femaleGenotype[x*2]
                            else:
                                haploF+=femaleGenotype[x*2+1]

                        else:
                            if int(z)==z:
                                haploF+=femaleGenotype[x*2+1]
                            else:
                                haploF+=femaleGenotype[x*2]
                    #/generate Maternal Allele to pass on


                    #intercolates the two haploid genotypes into the final genome for the new agent
                    egg.genotype=''        
                    for x in range(len(haploM)):
                        egg.genotype+=haploM[x] + haploF[x]
                    
                    if 'Y' in egg.genotype:
                        egg.sex='male'
                    else:
                        egg.sex='female'

                    #RULES FOR PROMOTER CONVERSION
                    if 'Pp' in egg.genotype or 'pP' in egg.genotype:
                        if random.random() < promoterConversionFreq:
                            egg.genotype = egg.genotype.replace('p','r')

                    if 'Bb' in egg.genotype or 'bB' in egg.genotype:
                        if random.random() < promoterConversionFreq:
                            egg.genotype = egg.genotype.replace('b','c')

                    if 'XU' in egg.genotype or 'UX' in egg.genotype:
                        if random.random() < promoterConversionFreq:
                            egg.genotype = egg.genotype.replace('X','u')

                    if 'p' in egg.genotype:
                        if random.random() < mutationRate:
                            egg.genotype = egg.genotype.replace('p','r')

                    if 'b' in egg.genotype:
                        if random.random() < mutationRate:
                            egg.genotype = egg.genotype.replace('p','r')
                    
                    if 'X' in egg.genotype:
                        if random.random() < mutationRate:
                            egg.genotype = egg.genotype.replace('X','x')
                    ##############################
                    egg.state = 'egglv'
                    egg.lifeStage = 1
                    if carry==True and self.tet==True:
                        egg.tet = 'Carry'
                    else:
                        egg.tet = False
                    self.model.grid.place_agent(egg, tuple(female.pos))
                    self.model.schedule.add(egg)
            

                self.model.num_agents = self.model.num_agents + Eggs2Lay       


        
        def female_mate(self):                                                  # Mating-ready females find random mating partner male
            cellmates = self.model.grid.get_cell_list_contents([self.pos])
            matingPartners = []
            if len(cellmates) > 1:
                for i in range(len(cellmates)):
                    if cellmates[i].sex == 'male' and 'adult' in cellmates[i].state: # Will mate with any adult male
                        matingPartners.append(cellmates[i])
                    else:
                        i=i+1
        
                if len(matingPartners) > 0:
                    matingPartner = random.choice(matingPartners)
                    self.mateGenotype.append(matingPartner.genotype)
                    if self.fertile == True and matingPartner.fertile == True:
                        matingPartner.mated = False                                 # males can mate multiple times
                        self.mated = True
                        self.pregnant = True
                        self.gestating = 1
                    else:
                        self.mated = False
                        self.pregnant = False
                        print("female_fertility; ",self.fertile, " Male_fertility: ",matingPartner.fertile)
                else:
                    return
            else:
                return    
            
        #defines actions taken by agent at each timestep
        def step(self):
            self.mature()
            if self.state =='adult' and self.sex == 'female':
                self.female_mate()
            if self.state =='adult' and self.sex == 'female' and self.mated == True :
                self.reproduce(self.mateGenotype, self)
            if random.random() < self.model.migration:
                self.move()
            
            else:
                self.mature()
            
    #CHANGE NAME
    class SWDModel(Model):
        """A model with some number of agents."""
        def __init__(self, N, width, height):
            self.num_agents = N
            self.grid = MultiGrid(width, height, True)
            self.schedule = RandomActivation(self)
            self.migration = 0
            self.zzz = 0
    
            for i in range(self.num_agents):
                a = SWD(i, self)
                self.schedule.add(a)
                
                # Add the agent to a random grid cell, this step is overridden if agent is placed to specific cell
                x = random.randrange(self.grid.width)
                y = random.randrange(self.grid.height)
                self.grid.place_agent(a,(x,y))
                
            self.datacollector = DataCollector(
                agent_reporters={"Sex": lambda a: a.sex,
                    "State": lambda a: a.state,
                    "Genotype": lambda a: a.genotype,
                    "Position": lambda a: a.pos})
        #step model takes with each timestep
        def step(self,timestep):
            '''Advance the model by one stop.'''               
            self.datacollector.collect(self)
            self.schedule.step()
            del TempData[0]
            print("Model step # ")
            
    
    #RunParameters
    
    filename = sys.argv[3]+typeExp + '_Release' + str(timeStepAdded)  + '.pkl'
    model = SWDModel(0,gridHeight,gridWidth)
    cells = gridWidth * gridHeight
    model.migration = migrationRate
    zzz=1                                                                       # zzz tracks number of agents, increases by one with each birth
    
                                                                                # put WT SWD in each cell
    for i in range(gridHeight):
        for j in range(gridWidth):
            for h in range(startingWT):
                new = SWD(model.zzz,model)
                new.state = 'adult'
                new.lifeStage = 0
                tempGenoType=wildtypeGenotype

                #introduaces SNPs at p or b allele in the Wildtype based on frequency defined earlier
                if random.random()<Freq1:
                    tempGenoType='r'+tempGenoType
                else:
                    tempGenoType='p'+tempGenoType
                if random.random()<Freq1:
                    tempGenoType='r'+tempGenoType
                else:
                    tempGenoType='p'+tempGenoType

                tempGenoType='dd'+tempGenoType

                #introduaces SNPs at p or b allele in the Wildtype based on frequency defined earlier
                if random.random()<Freq2:
                    tempGenoType='c'+tempGenoType
                else:
                    tempGenoType='b'+tempGenoType
                if random.random()<Freq2:
                    tempGenoType='c'+tempGenoType
                else:
                    tempGenoType='b'+tempGenoType

                if random.random()<sexRatio:
                    new.sex = 'female'
                else:
                    new.sex = 'male'
                if new.sex == 'female':
                    new.genotype = tempGenoType
                else:
                    new.genotype = tempGenoType[:-1]+"Y"

                new.mated = False
                new.pregnant = False
                new.mateGenotype = []
                new.tet=False
                model.grid.place_agent(new,tuple([i,j]))
                model.schedule.add(new)
                
            print("Seeded ",startingWT," wildtype in [", i, ",", j, "]")
    
        
                                                                                # Put GMO mosquitoes in center cells   
    for i in range(gridHeight-(2*borderCells)):
        for j in range(gridWidth-(2*borderCells)):
            for h in range(startingGMO):
                new = SWD(model.zzz,model)
            
                new.genotype = startingGMOgenotype
                if random.random()<GMsex:
                    new.genotype = startingGMOgenotype
                    new.sex='female'
                else:
                    new.genotype = startingGMOgenotype[:-1]+"Y"
                    new.sex='male'
                new.state = 'adult'
                new.tet = True
                new.lifeStage = 0
                model.grid.place_agent(new, tuple([i+borderCells,j+borderCells]))
                model.schedule.add(new)
            print("Seeded ",startingGMO," GMO in [", i+borderCells, ",", j+borderCells, "]")
    
    timeList=[]                                                                 # Model counter for debugging purposes
    with Timer('FULL EXECUTION'):            
        for i in range(timeSteps):


            # Adding new GE organisms and saving compressed datafile at each multiple of timestep added
            if i > 0 and i % timeStepAdded == 0:
                for j in range(gridHeight-(2*borderCells)):
                    for k in range(gridWidth-(2*borderCells)):

                        for l in range(addedGMO):

                            new =  SWD(model.zzz,model)
                            if random.random()<GMsex:
                                new.genotype = startingGMOgenotype
                                new.sex='female'
                            else:
                                new.genotype = startingGMOgenotype[:-1]+"Y"
                                new.sex='male'
                            new.state = 'adult'
                            new.tet = True
                            new.lifeStage = 0
                            model.grid.place_agent(new,tuple([j,k]))
                            model.schedule.add(new)
                            
                        print("Added ", addedGMO," GMO to in [", j, ",", k, "]")
                model.datacollector.get_agent_vars_dataframe()[['State','Genotype','Position', 'Sex']].copy().to_pickle(filename)
                with Timer('model step {}'.format(i)):
                    model.step(i)
            else:
                with Timer('model step {}'.format(i)):
                    model.step(i)
    
    gestatingCtr=0
    eggCtr=0
    elseCtr=0
    for a in model.schedule.agents:
        if a.state == 'gestating':
            gestatingCtr+=1
        elif a.state == 'egglv':
            eggCtr+=1
        else:
            elseCtr+=1
    print('gestating: {}'.format(gestatingCtr))
    print('egg: {}'.format(eggCtr))
    print('elseCtr: {}'.format(elseCtr))

    model.datacollector.get_agent_vars_dataframe()[['State','Genotype','Position', 'Sex']].copy().to_pickle(filename)
    if windows==True:
        winsound.Beep(5000,2000)

if windows==True:
    winsound.Beep(2000,2000)

'''
https://link.springer.com/article/10.1007/s10340-020-01292-w
https://www.mdpi.com/2075-4450/11/11/751/htm
https://link.springer.com/article/10.1007/s10340-015-0681-z
https://www.sciencedirect.com/science/article/pii/S0304380021002313#fig1
https://academic.oup.com/jee/article/109/2/746/2379864
https://academic.oup.com/ee/article/43/2/501/533184
'''

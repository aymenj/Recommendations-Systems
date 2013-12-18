# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
import numpy as np
import pandas as pd
from __future__ import division
import math

import matplotlib.pyplot as plt
import prettyplotlib as pplt
from prettyplotlib import plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (19, 11)

# <codecell>

critics = {'Aymen Jaffry': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5,
                         'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},
           'Yann Le Vacon': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0,
                             'The Night Listener': 3.0, 'You, Me and Dupree': 3.5},
           'Olivier Nicolas': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0, 'Superman Returns': 3.5,
                               'The Night Listener': 4.0},
           'Keunoo Chang': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Sarah Rossi': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'Just My Luck': 2.0, 'Superman Returns': 3.0,
                           'The Night Listener': 3.0, 'You, Me and Dupree': 2.0},
           'Igor Bernard': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0, 'The Night Listener': 3.0, 
                            'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Charles Gorintin': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}
           }

# <markdowncell>

# ###Similarity Scores
# 
# - **Euclidean Similarity**: works for quantitative and 0-1 values
# - **Pearson Similarity**: works for quantitative and 0-1 values. More efficient that the Euclidean score to identify when a user gives always a higher rating than the other user he is compared to
# - **Tanimoto Similarity**: for bit vectors (0-1 values)
# - **Manhattan/Taxicab**: Same as euclidean but with l1 measure

# <codecell>

##Euclidean Similarity
def sim_euclidean(prefs, person1, person2):
    dist = 0.
    for key in prefs[person1]:
        if key in prefs[person2]:
            dist += pow(prefs[person1][key] - prefs[person2][key],2)
    
    if dist==0: return 0.
    else: return 1./(1. + math.sqrt(dist))

##Pearson Similarity
def sim_pearson(prefs, person1, person2):
    count = 0.
    sum1 = 0.
    sum2 = 0.
    sum1Sq = 0.
    sum2Sq = 0.
    pSum = 0.
    for key in prefs[person1]:
        if key in prefs[person2]:
            count+=1
            sum1 += prefs[person1][key]
            sum2 += prefs[person2][key]
            sum1Sq += pow(prefs[person1][key],2)
            sum2Sq += pow(prefs[person2][key],2)
            pSum += prefs[person1][key]*prefs[person2][key]
            
    num = pSum - (sum1*sum2/count)
    den = math.sqrt((sum1Sq-pow(sum1,2)/count)*(sum2Sq-pow(sum2,2)/count))
    
    if den==0: return 0.
    else: return num/den
    
#Tanimoto Similarity
def sim_tanimoto(prefs,person1,person2):
    prod = 0.
    sum1Sq = 0.
    sum2Sq = 0.
    for key in prefs[person1]:
        if key in prefs[person2]:
            prod += prefs[person1][key] * prefs[person2][key]
            sum1Sq += pow(prefs[person1][key],2)
            sum2Sq += pow(prefs[person2][key],2)
            
    num = prod
    den = sum1Sq + sum2Sq - prod
    
    if den==0: return 0.
    else: return num/den
    
#Manhattan/Taxicab Similarity
def sim_manhattan(prefs,person1,person2):
    sum1 = 0.
    for key in prefs[person1]:
        if key in prefs[person2]:
            sum1 += math.fabs(prefs[person1][key] - prefs[person2][key])
    
    if sum1==0: return 0.
    else: return 1./(1.+sum1)

# <codecell>


# <codecell>

fig, ax = plt.subplots(4,2)
fig.subplots_adjust(hspace=.5,top = 1.5)
left = range(len(ticks))
for i,person in enumerate(critics):
    euclidean = []
    pearson = []
    manhattan = []
    for item in critics:
        if item != person:
            pearson.append(round(sim_pearson(critics,person, item),2))
            euclidean.append(round(sim_euclidean(critics,person, item),2))
            manhattan.append(round(sim_manhattan(critics,person, item),2))
    ticks = [item for item in critics]
    ticks.remove(person)
    pplt.bar(ax[i/2][i%2],left=left, 
                height=pearson, 
                annotate=True, 
                xticklabels=ticks,
                grid='y')
    ax[i/2][i%2].set_title(person.split(' ')[0])
plt.show()

# <codecell>

#Ranking the Users
def topMatches(prefs,person,n=5,similarity=sim_pearson):
    temp = [(similarity(prefs,person,item),item) for item in prefs if item!=person]
    temp.sort()
    temp.reverse()
    return temp[:n]

# <codecell>

topMatches(critics,'Charles Gorintin',n=5)

# <codecell>

#Recommend Movies
def getRecommendations(prefs, person, similarity=sim_pearson):
    weights = {}
    weighted_sums = {}
    for other in prefs:
        if other==person:
            continue
        sim = similarity(prefs,person,other)
        for item in prefs[other]:
            if item not in prefs[person] or prefs[person][item]==0:
                weighted_sums.setdefault(item,0)
                weighted_sums[item] += sim*prefs[other][item]
                weights.setdefault(item,0)
                weights[item] += sim
    
    rankings = [(total/weights[item],item) for item,total in weighted_sums.iteritems()]
    
    rankings.sort()
    rankings.reverse()
    return rankings

# <codecell>

getRecommendations(critics,'Charles Gorintin')

# <codecell>

##Find Similarity between items (and not user
def transformPrefs(prefs):
    temp = {}
    for user in prefs:
        for item in prefs[user]:
            temp.setdefault(item,{})
            temp[item][user] = prefs[user][item]
    
    return temp

# <codecell>

movies = transformPrefs(critics)
getRecommendations(movies,'Just My Luck')

# <codecell>

topMatches(movies,'Just My Luck',n=4,similarity=sim_manhattan)

# <codecell>



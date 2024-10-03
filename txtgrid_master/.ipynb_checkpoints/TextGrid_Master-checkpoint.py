#TODO map ipa to sampa

import argparse, copy
from collections import defaultdict
import wave
from os.path import basename, splitext, join, isfile
import bisect
import re
from difflib import SequenceMatcher
import numpy as np
from numba import jit
import logging
import pandas as pd


OFFSET = 0.01 #sec
DEVTH = 0.02

lTextRm = ('?','`',',','!','.')

@jit 
def is_sorted(a): 
    for i in range(a.size-1): 
        if a[i+1] < a[i] : 
           return False 
    return True 


def ReadWavFile(sWavFileName):
    if not isfile(sWavFileName):
        raise Exception(" Wave file {} not exist".format(sWavFileName))
    with wave.open(sWavFileName,'rb') as fWav:
        _wav_params = fWav.getparams()
        data = fWav.readframes(_wav_params.nframes)
    return _wav_params, data

def WriteWaveSegment(lCrnt_data, _wav_params, nFrams, sCrnt_wav_name):
    lWav_params = list(_wav_params)
    lWav_params[3] = nFrams
    with wave.open(sCrnt_wav_name,'wb') as fCrnt_wav:
        fCrnt_wav.setparams(tuple(lWav_params))
        fCrnt_wav.writeframes(lCrnt_data)
    return

def SplitWav(sWavFile, st, et, sOutName):
    _wav_params, data = ReadWavFile(sWavFile)
    start_byte = int(st * _wav_params.framerate * _wav_params.sampwidth)
    end_byte = int(et * _wav_params.framerate * _wav_params.sampwidth)
    st_indx = int(start_byte/2)*2
    ed_indx = int(end_byte/2)*2
    outData = data[st_indx:ed_indx]
    WriteWaveSegment(outData, _wav_params, int((ed_indx-st_indx)/_wav_params.sampwidth), sOutName)


def ParseTextTxtGrid(lLines):
    pItem = re.compile('(?<=item \[)([0-9]+)(?=\]:)')
    pTierSize = re.compile('(?<=intervals: size = )([0-9]*)')
    pPointSize = re.compile('(?<=points: size = )([0-9]*)')
    pTierName = re.compile('(?<=name = ")(.*)(?=")')
    pST = re.compile('(?<=xmin = )([-0-9\.]*)')
    pET = re.compile('(?<=xmax = )([-0-9\.]*)')
    pPNum = re.compile('(?<=number = )([-0-9\.]*)')
    pPMark = re.compile('(?<=mark = ")(.*)(?=")')
    pLabel = re.compile('(?<=text = ")(.*)(?=")')
    pTierType = re.compile('(?<=class = ")(.*)(?=")')
    
    dTiers = defaultdict(lambda: [[],[],[]])
    lTiers = []
    sCurLine = ''
    while 'tiers?' not in sCurLine and lLines:
        sCurLine = lLines.pop()
    if not lLines or 'exists' not in sCurLine:
        print('Bad format or no tiers')
        return
    nTiers = int(lLines.pop().split()[-1])
    for j in range(nTiers):
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')
                return
            _match = pItem.findall(sCurLine)
        iItemIndx = _match[0]
        
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')

            _match = pTierType.findall(sCurLine)
        sCTierType= _match[0]
        _match = None
        while not _match:
            try:
                sCurLine = lLines.pop()
            except IndexError:
                print('Bad format')
                return
            _match = pTierName.findall(sCurLine)
        sCTierName = _match[0]
        if sCTierType == 'IntervalTier':
            _match = None
            while not _match:
                try:
                    sCurLine = lLines.pop()
                except IndexError:
                    print('Bad format')
                    return
                _match = pTierSize.findall(sCurLine)
            nIntervals = int(_match[0])
            for i in range(nIntervals):
                sCurLine = lLines.pop()
                fST = float(pST.findall(lLines.pop())[0])
                fET = float(pET.findall(lLines.pop())[0])
                sLabel = pLabel.findall(lLines.pop())[0]
                dTiers[sCTierName][0].append(fST)
                dTiers[sCTierName][1].append(fET)
                dTiers[sCTierName][2].append(sLabel)
        elif sCTierType == 'TextTier':
            _match=None
            while not _match:
                try:
                    sCurLine = lLines.pop()
                except IndexError:
                    print('Bad format')

                _match = pPointSize.findall(sCurLine)
            nPoints = int(_match[0])
            
            for i in range(nPoints):
                sCurLine = lLines.pop()
                fST = float(pPNum.findall(lLines.pop())[0])
                fET = fST
                sLabel = pPMark.findall(lLines.pop())[0]
                dTiers[sCTierName][0].append(fST)
                dTiers[sCTierName][1].append(fET)
                dTiers[sCTierName][2].append(sLabel)
    return dTiers

def CompareTxtGrids(sTxtGrd1, sTxtGrd2, sTier1, sTier2, sMapFile, fDevThr = DEVTH):
    #load TxtGrid 1
    dTiers1 = ParseTxtGrd(sTxtGrd1)
    dTiers2 = ParseTxtGrd(sTxtGrd2)
    lST1, lET1, lLabels1 = dTiers1[sTier1]
    lST2, lET2, lLabels2 = dTiers2[sTier2]
    #Load phoneme map
    #print(lLabels1, lLabels2)
    if sMapFile:
        with open(sMapFile) as fMap:
            dMap = dict([l.strip().split() for l in fMap])
        for lLabels in lLabels1, lLabels2:
            for i in range(len(lLabels)):
                sLabel = lLabels[i].strip()
                if sLabel in dMap:
                    lLabels[i] = dMap[sLabel]
                elif sLabel:
                    print('{} not in MapFile'.format(sLabel))
    lJunks = ['sil','<p:>','','SIL']
    seq = SequenceMatcher(lambda x: x in lJunks,lLabels1,lLabels2)
    nPh1 = len([p for p in lLabels1 if p not in lJunks])
    nPh2 = len([p for p in lLabels2 if p not in lJunks])
    nMatchedPh = 0
    nDeviatedPh = 0
    lDevPhs = []
    lMatchedPhs = []
    #print(lLabels1,lLabels2)
    for a, b, size in seq.get_matching_blocks():
        #nMatchedPh += size
        for i in range(size):
            if lLabels1[a+i] not in lJunks:
                nMatchedPh += 1
                if abs(lST1[a+i] - lST2[b+i]) + abs(lET1[a+i] - lET2[b+i]) > fDevThr:
                    nDeviatedPh += 1
                    lDevPhs.append(lLabels1[a+i])
                lMatchedPhs.append(lLabels1[a+i])

    return (nPh1, nPh2, nMatchedPh, nDeviatedPh, lDevPhs, lMatchedPhs, seq.ratio())


#lTxtGrids --> is a list of textgrid file names ('txtGrid1','txtGrid2',...)
#sOutputFile --> name of merged text grid file
#sWavFile --> the path to the wav file of the textgrids, used to get the end time if not specified
#aSlctdTiers --> list of dict, each dict contains the requested tiers names in each textgrid file in lTxtGrids as keys and the new tier name as value [{tier1_source_name:tier1_dest_name,tier2_source_name:tier2_dest_name }, {tier1_source_name:tier1_dest_name,tier2_source_name:tier2_dest_name },...] Note: number of dicts should be same as number of TxtGrid files in lTxtGrids, if empty all tiers in all textgrids will be merged with thier original names
#aMapper: list of tubles each with name of tier and file to map labels to other symboles [('tier1','mapper file')]

def MergeTxtGrids(lTxtGrids, sOutputFile, sWavFile='', aSlctdTiers=[], aMapper = [], fST = None, fET = None):
    fST = 0.0 if not fST else fST
    if not fET:
        if not sWavFile:
            print('Specify either Ent time or path to wave file')
            return
        _wav_params, data = ReadWavFile(sWavFile)
        fET = _wav_params.nframes/_wav_params.framerate
    if not aSlctdTiers:
        aSlctdTiers = [{} for i in lTxtGrids]

    dTierMapper = dict(aMapper) if aMapper else None

    assert len(aSlctdTiers) == len(lTxtGrids)

    dMergTiers = defaultdict(lambda: [[],[],[]])

    i = 0

    for sTxtGrd, lSlctdTiers in zip(lTxtGrids,aSlctdTiers):
        dTiers = ParseTxtGrd(sTxtGrd)
        lSlctdTiers = {k:k for k in dTiers.keys()} if not lSlctdTiers else lSlctdTiers
        if dTierMapper:
            for sTier in lSlctdTiers:
                if sTier in dTierMapper:
                    with open(dTierMapper[sTier]) as fMap:
                        dMap = dict([l.strip().split() for l in fMap])
                    dTiers[sTier][2] = list(map(lambda x: dMap[x] if x in dMap else x,dTiers[sTier][2]))

        dMergTiers.update(dict([(lSlctdTiers[k],v) for k,v in dTiers.items() if k in lSlctdTiers]))
        #dMergTiers.update(dict([('{}'.format(k),v) for k,v in dTiers.items() if k in lSlctdTiers]))
        i = i+1

    WriteTxtGrdFromDict(sOutputFile, dMergTiers, fST, fET, sFilGab = '')
    return

#TextGrids should be unoverlaped
#TODO verify that intervals are not overlaped
def ConcatTxtGrids(lTxtGrids, tierNames = []):
    if not tierNames:
        print('Get tierNames from texgrid...')
        txtGrid = lTxtGrids[0]
        dTiers = ParseTxtGrd(txtGrid)
        tierNames = dTiers.keys()
    print('Merging {}'.format(', '.join([str(x) for x in tierNames])))
    dTiers = dict([(x,([],[],[])) for x in tierNames])
    for txtGrid in lTxtGrids:
        d = ParseTxtGrd(txtGrid)
        crntTiers = d.keys()
        assert set(crntTiers).issubset(tierNames), 'Missing tiers in file {}'.format(txtGrid)
        d = SortTxtGridDict(d)
        for tierName in crntTiers:
            for i in [0,1,2]:
                dTiers[tierName][i].extend(d[tierName][i])

    return SortTxtGridDict(dTiers)

def ParseChronTxtGrd(lLines):
    dTiers = defaultdict(lambda: [[],[],[]])
    dTiers_type = {}
    fST, fET = map(float,lLines.pop().split()[:2])
    nTiers = int(lLines.pop().split()[0])
    for i in range(nTiers):
        sTierType, sTierName = [x.strip('"') for x in lLines.pop().split()[:2]]
        dTiers_type[sTierName] = sTierType
    while lLines:
        sLine = lLines.pop()
        if sLine and sLine[0] == '!':
            sCTierName = sLine.split()[1].strip(':')
            lLine = lLines.pop().split()
            #sCTierName = lTiers[int(lLine[0])-1]
            if dTiers_type[sCTierName] == "IntervalTier":
                fST, fET = map(float,lLine[1:])
            elif dTiers_type[sCTierName] == "TextTier":
                fST = fET = float(lLine[1])
            sLabel = lLines.pop().strip('"')
            dTiers[sCTierName][0].append(fST)
            dTiers[sCTierName][1].append(fET)
            dTiers[sCTierName][2].append(sLabel)
    return(dTiers)

def ParseTxtGrd(sTxtGrd):
    with open(sTxtGrd) as fTxtGrd:
        lLines = fTxtGrd.read().splitlines()
    lLines.reverse()
    sCurLine = lLines.pop()
    if 'chronological' in sCurLine:
        dTiers = ParseChronTxtGrd(lLines)
    elif 'ooTextFile' in sCurLine:
        dTiers = ParseTextTxtGrid(lLines)
    else:
        print('TxtGrd format error or not supported')
        return
    return dTiers

#TODO use logging and raise error
def ValidateTextGridDict(dTiers, lSlctdTiers=[]):
    #Check if lSlctdTiers is in dTiers
    if lSlctdTiers:
        lNotIn = [i for i in lSlctdTiers if i not in dTiers.keys()]
        assert not lNotIn, '{} tiers not in dict'.format(' '.join(lNotIn))
    else:
        lSlctdTiers = dTiers.keys()
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        
        assert len(lSTs) == len(lETs), "Number of start times not equal number of end times in tier {}".format(sTier)
        
        assert len(lSTs) == len(lLabls), "Number of labels not equal to number of intervals in tier {}".format(sTier)

        aSTs = np.asarray(lSTs)
        aETs = np.asarray(lETs)

        assert (aETs >= aSTs).all(), "Start time is greater than end time in one or more intervals in tier {}".format(sTier)

        assert is_sorted(aETs) and is_sorted(aSTs), "Either End times or Start Times of tier {} not in order".format(sTier)

    return

"""
Merge all consecutive silence intervales, with labels ['sil','SIL','UNK','unk','']
Return same number of Tiers with selected tiers modified
"""
def Merge_sil(dTiers, lSlctdTiers=[], lSilWords=[], Replace=False):
    if not lSlctdTiers:
        lSlctdTiers = dTiers.keys()
    dOutTiers = dict([(x,([],[],[])) for x in dTiers.keys()])
    sil_words = ['sil','SIL','UNK','unk','<p:>']
    if lSilWords:
        sil_words = lSilWords if Replace else sil_words+lSilWords
    sil_symbol = 'sil'
    for Tier in dTiers.keys():
        if Tier in lSlctdTiers:
            tmp = []
            for fST,fET,sLabel in zip(*dTiers[Tier]):
                if sLabel in sil_words:
                    tmp.append((fST,fET,sLabel))
                else:
                    if tmp:
                        dOutTiers[Tier][0].append(tmp[0][0])
                        dOutTiers[Tier][1].append(tmp[-1][1])
                        dOutTiers[Tier][2].append('sil')
                        tmp = []
                    dOutTiers[Tier][0].append(fST)
                    dOutTiers[Tier][1].append(fET)
                    dOutTiers[Tier][2].append(sLabel)
        else:
            dOutTiers[Tier] = dTiers[Tier]
    return dOutTiers

"""
Merge all consecutive non-silence intervales, useful in word tiers to create trans, if sil interval < min_sil remove it
Add merged label tier for each selected tier
"""
def Merge_labels(dTiers, lSlctdTiers=[], min_sil_dur=0.2):
    sil_words = ['sil','SIL','UNK','unk','<p:>','']
    if not lSlctdTiers:
        lSlctdTiers = dTiers.keys()
    dOutTiers = copy.copy(dTiers)
    for Tier in lSlctdTiers:
        new_tier = 'm-{}'.format(Tier)
        dOutTiers[new_tier] = ([],[],[])
        fSTn = -1
        sLabeln = ''
        for fST,fET,sLabel in zip(*dOutTiers[Tier]):
            if sLabel in sil_words:
                if fET-fST >=min_sil_dur:
                    if fSTn !=-1:
                        fETn = fST
                        dOutTiers[new_tier][0].append(fSTn)
                        dOutTiers[new_tier][1].append(fETn)
                        dOutTiers[new_tier][2].append(sLabeln)
                        fSTn = fETn = -1
                        sLabeln = ''
                    dOutTiers[new_tier][0].append(fST)
                    dOutTiers[new_tier][1].append(fET)
                    dOutTiers[new_tier][2].append(sLabel)
            else:
                if fSTn == -1:
                    fSTn = fST
                sLabeln = sLabeln + ' ' + sLabel
    return dOutTiers





def FillGapsInTxtGridDict(dTiers,sFilGab = "", lSlctdTiers=[]):
    """
    Should be called always after ValidateTextGridDict to make sure that the dTiers is valid
    """
    dTiers_f = defaultdict(lambda: [[],[],[]])
    lSlctdTiers = lSlctdTiers if lSlctdTiers else dTiers.keys()
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        aSTs = np.asarray(lSTs)
        aETs = np.asarray(lETs)
        aLabels = np.asarray(lLabls,dtype=str)
        pos = np.where((aSTs[1:] - aETs[:-1]) > 0.00001)
        lSTs_f = np.insert(aSTs,pos[0]+1,aETs[pos[0]])
        lETs_f = np.insert(aETs,pos[0]+1,aSTs[pos[0]+1])
        lLabel_f = np.insert(aLabels,pos[0]+1,sFilGab)
        dTiers_f[sTier] = [lSTs_f,lETs_f,lLabel_f]
    return dTiers_f

def SortTxtGridDict(dTiers):
    dTiers = copy.copy(dTiers)
    for p in dTiers:
        lSTs, lETs, lLabls = dTiers[p]
        aSTs = np.asarray(lSTs)
        aETs = np.asarray(lETs)
        aLabels = np.asarray(lLabls)
        indxSort = np.argsort(aSTs)
        dTiers[p] = (aSTs[indxSort], aETs[indxSort],aLabels[indxSort])

    return dTiers

def WriteTxtGrdFromDict(sFName, dTiers, fST, fET, bReset=True, lSlctdTiers=[], sFilGab = None, bRemoveOverlap=True):
    ValidateTextGridDict(dTiers,lSlctdTiers)
    if sFilGab != None:
        dTiers = FillGapsInTxtGridDict(dTiers,sFilGab,lSlctdTiers)
    if bRemoveOverlap:
        dTiers = RemoveOverlapIntervals(dTiers)
 
    fST = round(fST,4)
    fET = round(fET,4)
    if bReset:
        fNewST = 0
        fNewET = fET - fST
    else:
        fNewST = fST
        fNewET = fET
    lAllIntervals = []
    lSlctdTiers = [i for i in lSlctdTiers if i in dTiers.keys()]
    if not lSlctdTiers:
        lSlctdTiers = list(dTiers.keys())
    for sTier in lSlctdTiers:
        lSTs, lETs, lLabls = dTiers[sTier]
        lSTs = [round(i,4) for i in lSTs]
        lETs = [round(i,4) for i in lETs]
        iSindx = bisect.bisect_left(lSTs, fST)
        iEindx = bisect.bisect(lETs, fET)+1
        #print(iSindx,iEindx)
        #Adjust start and end times
        #print(lSTs,lETs)
        lNewSTs = lSTs[iSindx:iEindx]
        lNewETs = lETs[iSindx:iEindx]
        lNewLabls = lLabls[iSindx:iEindx]
        #print(lNewSTs,lNewETs)
        # TODO consider not to make same start and end time for all
        #Add sil interval at the start and end
        #lNewSTs = [fST] + lNewSTs + [lNewETs[-1]]
        #lNewETs = [lNewSTs[0]] + lNewETs + [fET]

        # Reset to 0
        if bReset and fST != 0.0:
            lNewSTs = [i-fST for i in lNewSTs]
            lNewETs = [i-fST for i in lNewETs]
        try:
            if fNewST < lNewSTs[0]:
                lAllIntervals.append((fNewST, lNewSTs[0], 'sil' ,sTier))
        except:
            print('Error:', lNewSTs, lNewETs, iSindx, iEindx)
            return
        for fiST, fiET, siLabl in zip(lNewSTs,lNewETs,lNewLabls):
            lAllIntervals.append((fiST, fiET, siLabl,sTier))
        if fNewET > lNewETs[-1]:
            #print(round(fNewET,3),round(lNewETs[-1],3),sTier)
            lAllIntervals.append((lNewETs[-1], fNewET, 'sil' ,sTier))

    lAllIntervals = sorted(lAllIntervals)
    # Writing the chron txtgrid
    with open(sFName,'w') as fTxtGrd:
        print('"Praat chronological TextGrid text file"', file=fTxtGrd)
        print('{} {}   ! Time domain.'.format(fNewST, fNewET),file=fTxtGrd)
        print('{}   ! Number of tiers.'.format(len(lSlctdTiers)), file=fTxtGrd)
        for sTier in lSlctdTiers:
            print('"IntervalTier" "{}" {} {}'.format(sTier,fNewST,fNewET), file=fTxtGrd)
        print('', file=fTxtGrd)
        for fiST, fiET, siLabl, sTier in lAllIntervals:
            indx = lSlctdTiers.index(sTier) + 1
            print('! :{}'.format(sTier), file=fTxtGrd)
            print('{} {} {}'.format(indx, fiST, fiET), file=fTxtGrd)
            print('"{}"'.format(siLabl), file=fTxtGrd)
            print('', file=fTxtGrd)
    return        

def subTxtgrid(dTiers, TrgtST, TrgtET, lTiers=[]):
    if not lTiers:
        lTiers = dTiers.keys()
    dOutTiers = dict([(x,([],[],[])) for x in lTiers])
    for tier in lTiers:
        for st, et, label in zip(*dTiers[tier]):
            if st > TrgtST and et < TrgtET:
                dOutTiers[tier][0].append(st)
                dOutTiers[tier][1].append(et)
                dOutTiers[tier][2].append(label)
    return dOutTiers

def TextNormalize(sTxt):
    pNorm = re.compile('\(.*?\)|'+'['+re.escape(''.join(lTextRm))+']')
    return pNorm.sub('',sTxt)

def Process(sTxtGrd, sWavFile, sSplitBy, sOutputDir, bTxtGrd = False, bNorm = True):
    dTiers = ParseTxtGrd(sTxtGrd)
    _wav_params, data = ReadWavFile(sWavFile)
    sBaseName = splitext(basename(sWavFile))[0]
    for fST, fET, sLabel in zip(*dTiers[sSplitBy]):
        if sLabel:
            print('File: {} - time {} to {} - label {}'.format(sTxtGrd,fST, fET, sLabel))
            indxFS = int((fST - OFFSET) * _wav_params.framerate) #start one frame before
            indxFE = int((fET + OFFSET) * _wav_params.framerate) #end one frame after
            nFrams = indxFE - indxFS
            lCrnt_data = data[indxFS * _wav_params.sampwidth:indxFE * _wav_params.sampwidth]
            sCrnt_name = "{0}_{1:.2f}_{2:.2f}".format(sBaseName,fST,fET)
            cCrnt_wav_name = join(sOutputDir,"{0}.wav".format(sCrnt_name))
            sCrnt_txt_name = join(sOutputDir,"{0}.txt".format(sCrnt_name))
            WriteWaveSegment(lCrnt_data, _wav_params, nFrams, cCrnt_wav_name)
            with open(sCrnt_txt_name,'w') as fCrnt_txt:
                if bNorm:
                    sLabel = TextNormalize(sLabel)
                print(sLabel,file=fCrnt_txt)
            if bTxtGrd:
                sCrnt_txtGrd_name = join(sOutputDir,"{0}.textgrid".format(sCrnt_name))
                fST = indxFS/_wav_params.framerate #Recalculate the time after OFFSET
                fET = indxFE/_wav_params.framerate
                WriteTxtGrdFromDict(sCrnt_txtGrd_name, dTiers, fST, fET, bReset=True, lSlctdTiers=[])
    return

def RemoveOverlapIntervals(dTiers):
    dTiers_fixed = {}
    for tierName in dTiers:
        df = pd.DataFrame.from_dict({'st':dTiers[tierName][0],'et':dTiers[tierName][1],'label':dTiers[tierName][2]})
        for i in range(df.shape[0]-1):
            if df.loc[i].et > df.loc[i+1].st:
                df.loc[i,'et'] = df.loc[i+1].st
        dTiers_fixed[tierName] = [df.st.values, df.et.values, df.label.values]
    return dTiers_fixed

def dictToDataFrame(dTiers):
    df_dict = {'from':[],'to':[],'label':[],'tier':[]}
    for tier_name in dTiers:
        nRecords = len(dTiers[tier_name][0])
        df_dict['from'] = df_dict['from'] + dTiers[tier_name][0]
        df_dict['to'] = df_dict['to'] + dTiers[tier_name][1]
        df_dict['label'] = df_dict['label'] + dTiers[tier_name][2]
        df_dict['tier'] = df_dict['tier'] + [tier_name]*nRecords
    df_tiers = pd.DataFrame.from_dict(df_dict)
    return(df_tiers)

def ArgParser():
    parser = argparse.ArgumentParser(description='This code split wav based on textgrid alignment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('TxtGrd',  help='The path to the TextGrid file', type=str)
    parser.add_argument('WavFile',  help='The path to the associated wav file', type=str)
    parser.add_argument('SplitBy',  help='The tier name to split the wav file based on', type=str)
    parser.add_argument('OutputDir',  help='The path to the output dir', type=str)
    parser.add_argument('-t','--txtgrd', help='Use this option to output TextGrid file for each segment', dest='txtgrd_o', action='store_true',default=False)
    parser.add_argument('-s','--spontaneous', help='Use this option to enable processing of spontaneous part of the data', dest='process_spontaneous_data', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = ArgParser()
    sTxtGrd, sWavFile, sSplitBy, sOutputDir = args.TxtGrd, args.WavFile, args.SplitBy, args.OutputDir
    bTxtGrd = args.txtgrd_o
    Process(sTxtGrd, sWavFile, sSplitBy, sOutputDir, bTxtGrd = bTxtGrd)




if __name__=='__main__':
    main()

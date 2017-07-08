#!/sw/bin/python3.3
#! E:\Python33\python
#Read FASTA files from current directory, generate output txt file with values for features.
#30.6.2013 . Edited order of features generated, and bigrams from absolute to relative freq. + added length

#import Bio
#added features not yet put in featuregen classic
import re
import numpy as np
import sys
from itertools import permutations
from itertools import product
from Bio import SeqIO
from Bio.SeqUtils import IsoelectricPoint
from Bio.SeqUtils import ProtParam as pp
import random
import os
from math import log
from collections import defaultdict
import numba
from numba import autojit

#parameters that change depending on + or - set - output file's name + number of samples. (Neg - 120 samples each)
outPut_filename = "Features_POS+"
num_samples = 900
negative_set = False
unwanted_residues = ['U','X','Z','B']
#Total Np+, 90% redundnacy, NOT receptors (10.10.2012) = 627. 84% = 530
k = raw_input ('Positive (+) or Negative (-) training set? \n "-" for negative, any other sign for "+" \n')
if k == '-':
   outPut_filename = "Features_NEG-"
   num_samples = 190
   negative_set = True


#Used tofilter out ,U,B,Z,X non standrd AAs from a given sequence/string. Returns true if illegals present
#@jit
def contain_illegals(str, set):
   for c in set:
      if c in str:
         print("ILLEGAL AA!")
         return True
    return False;

# Data is Imported from a FASTA sequence file:
# list - each entry is appended strings/the sequence
def parse_fasta(filename) :
   #f = open(sys.argv[1]+'/'+filename, 'r')
   f = open(filename, 'r')
   sequences =[]
   i = 0
   for line in f:
         if line.startswith('>'):
            i = (i+1)
         else:
            if not contain_illegals(line,unwanted_residues):
               if (len(sequences) - 1 == i):
                  sequences[i] += line.rstrip('\n')
               else:
                  sequences.append(line.rstrip('\n'))
   return sequences

#Writes out results to (same file each time) file with name "outname".txt, param is the key/values dict to print
#Modify: - 'for Key in dictionary, write out k[V], /n .... (after iterating over all key's values, close file)
## This loop syntax accesses the whole dict by looping over the .items() tuple list
#  for k, v in dict.items(): print k, '>', v
def outWrite(param, outName) :
   out = open('./'+outName + '.txt', "w")
   for k, v in param.items() :
      out.write('\t'.join(map(str, v)))
      out.write('\n')
      #print(' Values: \n' + ('\t'.join(map(str, v))))
   out.close()
   return


def getStr(param):
    data = ""
    for k, v in param.items() :
        data += '\t'.join(map(str, v))
        data += '\n'

    return data


@jit
def KROne(seq) :
   seq = re.sub("\D", '0', seq)
   return seq
# GKR = 1 (Glycine)
@jit
def GKROne(seq) :
   seq = seq.replace('G', '1')
   seq = re.sub("\D", '0', seq)
   return seq
#Hydrophibicity - "Charged AA" (DERHK) = '1'
@jit
def chargeOne(seq) :
   seq = seq.replace('D', '1').replace('E', '1').replace('H', '1')
   seq = re.sub("\D", '0', seq)
   return seq

#@jit
def MerCount(s) :
#Combo [list] holds all the 2^5 binary combinations
#Transpose list of permutations (combo) into a new defaultdictionary (key,0)!
   d = dict.fromkeys(combo, 0)
   for i in xrange(len(s) - 4) :
      d[s[i :i + 5]] += 1
   return d.values()

# a and b are  relative volume of valine and Leu/Ile side chains to side chain of alanine.
# http://stackoverflow.com/questions/991350/counting-repeated-characters-in-a-string-in-python
@jit
def aliphaticness(seq) :
   a = 2.9
   b = 3.9
   length = float(len(seq))
   alanine_per = (seq.count('A') / length )
   valine_per = (seq.count('V') / length )
   isoleucine_per = (seq.count('I') / length )
   leucine_per = (seq.count('L') / length )
   # Aliphatic index = X(Ala) + a * X(Val) + b * ( X(Ile) + X(Leu) )
   aliphatic_index = (100 * (alanine_per + a * valine_per + b * (isoleucine_per + leucine_per )))
   return aliphatic_index

@jit
def Autocorrellation(seq,loc) :
   seq = seq.replace('RR', '1').replace('KK', '1').replace('RK', '1').replace('KR', '1')
   seq = seq.replace('R', '1').replace('K', '1')
   seq = re.sub("\D", '0', seq)
   seq = map(int, seq)
   selfCor = np.correlate(seq, seq, 'full')
   #Added to Avoid divide by zero error:
   if sum(seq)==0:
      return 0
   #Normalization - By Sum ("1's") or seq.length?
   autoCor = sorted(selfCor)[loc] / float(len(seq))
   #Second highest (NOT "100%" Autocorrelation : loc=-2
   return autoCor


# Counts percentage of occurences of biGrams (from bigramDict) for  a given seq
def bigramsFreq(seq, bigramDict ) :
   length=((len(seq))/2)
   for Aa in bigramDict.keys() :
      bigramDict[Aa] = ((seq.count(str(Aa)))/length)
      #print bigramDict
   return bigramDict.values()


def seq_Entropy(seq) :
   length = float(len(seq))
   letters = list('ACDEFGHIKLMNPQRSTVWY')
   amino_acids = dict.fromkeys(letters, 0)
   for Aa in amino_acids :
      hits = []
      hits = [a.start() for a in list(re. finditer(Aa, seq))]
      p_prev = 0
      p_next = 1
      sum = 0
      while p_next < len(hits) :
         distance = (hits[p_next] - hits[p_prev]) / length
         sum += distance * log(distance, 2)
         p_prev = p_next
         p_next += 1
      amino_acids[Aa] = -sum
   return amino_acids.values()


#Code for Finding AA motifs counts
def PrefixCount(Aa, seq) :
   counts = len(re.findall('[%s].{0,1}[KR][KR]' % (Aa), seq))
   return counts / float(len(seq))

def SuffixCount(Aa, seq) :
      counts = len(re.findall('[KR][KR].{0,1}[%s]' % (Aa), seq))
      return (counts / float(len(seq)))

def NSites (seq,length):
	''' N Glycosylation sites'''
	NSites = len(re.findall(r'N[^P][ST][^P]', seq))
	return (NSites/length)

# Aspartic acid, asparagine hydroxylation sites
def hydroxSites (seq,length):
   hydroxSites = len(re.findall('CC.{13}C.{2}[GN].{12}C.C.{2, 4}C',seq))
   return (hydroxSites/length)

   #counts # of suspected cleavage sites according to known motif model
   # Xxx-Xxx-Lys-Lys# , Xxx-Xxx-Lys-Arg# , Xxx-Xxx-Arg-Arg# ,  Arg-Xxx-Xxx-Lys# , # # Arg-Xxx-Xxx-Arg#
   # lysine: K. arginine: R.   #"Not Proline" = [^P]
@jit
def cleavageCounts(seq) :
   count1 = len(re.findall('R.[^P][RK]', seq))
   #Arg-Xxx-Xxx-Arg|Lys
   count2 = (len(re.findall('.[^P][RK][RK]', seq)))
   return (count1 + count2)

#@jit
# Bigramsdict
def gen_BigramDict():
    bigramsAll = []
    for i in (permutations('ACDEFGHIKLMNPQRSTVWY', 2)) :
       bigramsAll.append(i[0] + i[1])
    bigramDict = dict.fromkeys(bigramsAll, 0)
    return bigramDict
# bigramDict = dict containing all the valid 2 letter bigrams as keys


#==============================================================================
# #main CODE:
#==============================================================================
combo = [''.join(x) for x in product('01', repeat=5)] #combo = list of all  possible [0/1] ,length 5 combinations
sampled_proteins = {}
sampled_seq = []
sequences = {} #will contain sequences as values (rather than JUST as "keys" as is case with 'sampled_proteins')
aa_groups = ('FYW', 'P', 'C', 'RHK', 'DE' , 'CSTMNQ','RK',
'ST','LASGVTIPMC','EKRDNQH')
bigramDict=gen_BigramDict()

#Read FASTA files from current directory
#for f in os.listdir(sys.argv[1]) :
files = [f for f in os.listdir(os.curdir) if (os.path.isfile(f) and f.endswith(".fasta"))]
for f in files:
#for f in os.listdir(os.curdir) :
   if (negative_set): #If negative sets marker set to true due to user input
      if f.endswith(".fasta") and not f.startswith("_") and f.startswith('NEG') or f.startswith('Neg') :
         Fasta_seq = parse_fasta(f)
#num_samples = How many samples to sample at random from each file (in the given directory).
         sampled_seq += random.sample(Fasta_seq, num_samples)
         sampled_proteins = dict.fromkeys(sampled_seq,0)
         #NP+ Positive  case
   elif f.endswith(".fasta") and not f.startswith("_"):
      Fasta_seq = parse_fasta(f)
      # sampled_seq += random.sample(Fasta_seq, len(Fasta_seq))
      #Make samples seqs for NP+ Contain all the NP+:
      sampled_seq +=Fasta_seq
      sampled_proteins = dict.fromkeys(sampled_seq,0)
   sequences = dict(zip(sampled_seq, sampled_seq))

#http://biopython.org/wiki/ProtParam
for seq in sampled_proteins :
   length = float(len(seq))
   Z = pp.ProteinAnalysis(sequences[seq].replace('X', '').replace('Z', ''))
   sampled_proteins[seq] = []
   window_mer = sequences[seq].replace('R', '1').replace('K', '1')

   sampled_proteins[seq].append(length)
   sampled_proteins[seq].append(Z.isoelectric_point())
   sampled_proteins[seq].append(Z.molecular_weight())
   sampled_proteins[seq].append(Z.gravy())
   sampled_proteins[seq].append(Z.aromaticity())
   sampled_proteins[seq].append(Z.instability_index())
   # (Z.flexibility())
   #protparam AA% returns a dict. of K(Aa):V(%) pairs
   sampled_proteins[seq].append(Autocorrellation(sequences[seq],-2))
   #sampled_proteins[seq].append(Autocorrellation(sequences[seq],-3))
   sampled_proteins[seq].append(aliphaticness(sequences[seq]))

   # N Glycosylation sites
   sampled_proteins[seq].append(NSites(sequences[seq],length))
   # Aspartic acid, asparagine hydroxylation sites
   sampled_proteins[seq].append(hydroxSites(seq,length))

   #Counts of suspected cleavage sites
   sampled_proteins[seq].append(cleavageCounts(sequences[seq]) / length)

   sampled_proteins[seq].extend(Z.get_amino_acids_percent().values())
   sampled_proteins[seq].extend(MerCount(KROne(window_mer)))
   sampled_proteins[seq].extend(MerCount(GKROne(window_mer)))
   sampled_proteins[seq].extend(MerCount(chargeOne(window_mer)))
   sampled_proteins[seq].extend(seq_Entropy(sequences[seq]))
      #AA Bigrams (400) frequencies:
   #sampled_proteins[seq].extend(bigramsFreq(sequences[seq], bigramDict))
   for Aa in aa_groups :
      sampled_proteins[seq].append(PrefixCount(Aa, sequences[seq]))
      sampled_proteins[seq].append(SuffixCount(Aa, sequences[seq]))
#Finally Write out results (with seperate lines per key/sequence) to a new file:
outWrite(sampled_proteins, outPut_filename)

#print 'length'
#print len(sampled_proteins)
print 'Done'

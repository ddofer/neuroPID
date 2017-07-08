#!/sw/bin/python2.7
import urllib,urllib2, os
lengths= [(8,24),(25,45),(46,70),(71,91),(92,110),(111,150),(151,190),(191,220),(221,300),(301,370),(371,450),(451,600),(601.800),(801,1200),(1201,3000),(3001,4703)]

#updated for "fragment%3Ano", 0.5id?
for aRange in lengths:
    urlNotNeuroPeps ='http://www.uniprot.org/uniref/?query=not+annotation:(type:non_std)+AND+length:[%s+TO+%s]+AND+reviewed:yes+AND+fragment%3Ano+AND+identity:0.5+NOT+keyword:"Neuropeptide+KW-0527"&format=fasta'  %(str(aRange[0]), (str(aRange[1])))
    os.system('wget \'%s\' -O /cs/prt3/danofer/FEATURECode/+-Sets/NEG_%s.fasta' % (urlNotNeuroPeps,str(aRange[0])))

#Get All NPs (FASTA):

#updated for "fragment%3Ano"
os.system('wget \'%s\' -O /cs/prt3/danofer/FEATURECode/+-Sets/NP+.fasta'
%('http://www.uniprot.org/uniref/?query=uniprot%3a(neuropeptide+AND+reviewed%3ayes+NOT+keyword%3areceptor+AND+fragment%3Ano)+identity%3a0.9&force=yes&format=fasta'))

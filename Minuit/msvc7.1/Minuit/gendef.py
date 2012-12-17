#!/usr/bin/env python2
import sys, string, os

#----------------------------------------------------------------------------------
def genfile( fname ) :
  s = ''
  command = 'dumpbin/symbols '+ fname 
  for line in os.popen(command).readlines() :
    if line.find('SECT') == -1 : continue 
    words = line.split()
    if 'External' in words :
      symbol = words[words.index('|')+1]
      if symbol.find('??_') != -1 : continue
      if symbol[0:5] == '_SEAL' : symbol = symbol[1:]
      if '()' not in words : s += '  '+ symbol + '\tDATA\n'
      else :                 s += '  '+ symbol + '\n'
  return s
  
#----------------------------------------------------------------------------------
def usage() :
  print 'Usage:'
  print '  gendef [object/library file] [options]'
  print 'Try "gendef --help" for more information.'
  sys.exit()
#----------------------------------------------------------------------------------
def help() :
  print """Generates Module-Definition (.def) File\n
  print 'Usage:'
  print '  gendef [object/library file] [options]'
  Options:
     -h, --help
       Print this help\n
     -o, --output=
       Output file\n 
   """ 
  sys.exit()
#----------------------------------------------------------------------------------
def main() :
  import getopt
  if len(sys.argv) < 2 :
    print 'No object/library file as input'
    usage()
  if sys.argv[1][0] == '-' :
    infile  = None  
    options = sys.argv[1:]
  else :
    infile  = sys.argv[1]
    options = sys.argv[2:]
  output = None;
  #----Process options--------------------------------
  try:
    opts, args = getopt.getopt(options, 'ho:', ['help','output='])
  except getopt.GetoptError:
    usage()
  for o, a in opts:
    if o in ('-h', '--help'):
      help()
    if o in ('-o', '--output'):
      output = a
  #--------------------------------------------------------------------
  if output : 
      fo = open(output,'w')
      library = string.split (output, '.' )[0]
      library = os.path.split(library)[1]
  else      : 
       fo = sys.stdout
       library = 'tmp'
#  library = os.path.split(infile)[1]
  header  = 'LIBRARY '+ library + '\n'
  header += 'EXPORTS'+'\n'
  fo.write(header)
  if os.path.isdir(infile) :
    for f in os.listdir(infile) :
      if f[f.find('.'):] not in ('.obj','.lib') : continue
      fo.write( genfile(infile+'\\'+f) )  
  elif os.path.isfile(infile) :
    fo.write( genfile(infile) )
  

#---------------------------------------------------------------------
if __name__ == "__main__":
  main()


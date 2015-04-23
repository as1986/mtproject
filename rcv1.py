#!/usr/bin/python

def main():
   from xml.dom.minidom import parseString
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('inf')
   args = parser.parse_args()
   fh = open(args.inf, mode='r')
   l = fh.read()
   fh.close()
   parsed = parseString(l)
   t0 = parsed.getElementsByTagName('text')[0]
   p0 = t0.getElementsByTagName('p')
   for p in p0:
      print p.firstChild.nodeValue

if __name__ == '__main__':
   main()

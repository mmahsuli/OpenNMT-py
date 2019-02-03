# Author: Mohammad Mahdi Mahsuli
# Show length stats of the input file
import sys
import statistics as s
file_name=sys.argv[-1]

print('counting...')
f = open(file_name, 'r')
lens=[]
while True:
	line = f.readline() # src
	if not line:
		break;	
	lens.append(len(line.split()))
print('line count: '+str(len(lens)))
print('Min: '+str(min(lens)))
print('Max: '+str(max(lens)))
print('Mean: '+str(s.mean(lens)))
print('StDev: '+str(s.stdev(lens)))
f.close()



#This script aligns the predicted answer with the correct answers and stores the predicted answer in "pred.txt" and correct answer in "actual.txt"
#The predicted and correct answer is then used in calculate WUPS score script.  

import sys
import os
import re

if __name__ == '__main__':
	with open('qans.txt') as f:
		content = f.readlines()
	questions = {}

	length = len(content)/2
	for index in range(0, length) :
		q = content[2*index].strip()
		a = content[2*index+1].strip()
		match = re.search(r'image\d+', q)
		if match.group(0) in questions:
			questions[match.group(0)].append({'q' : content[2*index], 'a': a })
		else:
			questions[match.group(0)] = list() 
	print questions


	with open('train_8000googlenet.txt') as f1:
		output = f1.readlines()
	

	actual = open('actual.txt', 'w')
	pred = open('pred.txt', 'w')
	
	length = len(output)/3
	for index in range(0, length) :
		im = output[3*index+2]
		image = im.split('/')[1].strip().split('.')[0]
		q = output[3*index].strip().split("?")[0].strip().split(':')[1]
		for val in questions[image]:
			quest = val['q'].strip()
			if re.match(q, quest):
				# print q, quest
				actual.write(val['a'].strip()+'\n')
				pred.write(output[3*index+1].strip().split(':')[1]+'\n')
	actual.close()
	pred.close()

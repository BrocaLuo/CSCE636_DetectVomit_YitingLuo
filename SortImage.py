import os
DIRECTORY= "C:\\Vomit\\Sort\\NOV"
f = open('C:\\Vomit\\Sort\\NOV\\NOV.txt','w')
files=os.listdir(DIRECTORY)
for file in files:
		f.writelines(file+" "+'0')
		f.write('\n')
f.close()

DIRECTORY= "C:\\Vomit\\Sort\\V"
f = open('C:\\Vomit\\Sort\\V\\V.txt','w')
files=os.listdir(DIRECTORY)
for file in files:
		f.writelines(file+" "+'1')
		f.write('\n')
f.close()

DIRECTORY= "C:\\Vomit\\Test\\0"
f = open('C:\\Vomit\\Test\\0\\0.txt','w')
files=os.listdir(DIRECTORY)
for file in files:
		f.writelines(file+" "+'0')
		f.write('\n')
f.close()

DIRECTORY= "C:\\Vomit\\Test\\1"
f = open('C:\\Vomit\\Test\\1\\1.txt','w')
files=os.listdir(DIRECTORY)
for file in files:
		f.writelines(file+" "+'1')
		f.write('\n')
f.close()


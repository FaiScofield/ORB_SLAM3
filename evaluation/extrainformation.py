#fi = open("odom.txt","r",encoding="utf-8")
#fo = open("information.txt","w",encoding="utf-8")

import io
fi = io.open("origodom.txt", "r", encoding='utf-8')
fo = io.open("odom.txt","w",encoding="utf-8")

newline = ''
preline = ''
prepreline = ''
preprepreline = ''

for line in fi:
    if 'header' in line:
        newline += '\n'
    if 'secs: ' in line and 'nsecs' not in line:
        newline += line.strip()
	newline = newline.replace("secs: ","")
        #newline += '.'
    if 'nsecs: ' in line:
	line = line.replace("nsecs: ","")
	#line = line.replace(" ","")
	line = float((float(line.lstrip()))/1000000000);
	line = str(line).lstrip('0')
        newline += line
        newline += '    '

    if 'x' in line and preline == 'position:':
        newline += line.strip()
	newline = newline.replace("x: ","")
        newline += '    '
    if 'y' in line and prepreline == 'position:':
        newline += line.strip()
	newline = newline.replace("y: ","")
        newline += '    '
    if 'z' in line and preprepreline == 'position:':
        newline += line.strip()
	newline = newline.replace("z: ","")
        newline += '    '

    if 'x' in line and preline == 'orientation:':
        newline += line.strip()
	newline = newline.replace("x: ","")
        newline += '    '
    if 'y' in line and prepreline == 'orientation:':
        newline += line.strip()
	newline = newline.replace("y: ","")
        newline += '    '
    if 'z' in line and preprepreline == 'orientation:':
        newline += line.strip()
	newline = newline.replace("z: ","")
        newline += '    '
    if 'w:' in line:
        newline += line.strip()
	newline = newline.replace("w: ","")

    preprepreline = prepreline.strip()
    prepreline = preline.strip()
    preline = line.strip()

#print(newline)
fo.write(newline)

fo.close()
fi.close()


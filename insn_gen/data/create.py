file = open('instruction_dump', mode='r')
lines = file.read().splitlines()
file.close()

import random

tests=[]
lhs=0
while True:
	# print(lhs)
	count=random.randint(60,100)
	testlet=[]
	if lhs+count>=len(lines)-1:
		testlet.append(';'.join(lines[lhs:]))	
		tests.append(testlet)
		break
	else:
		testlet.append(';'.join(lines[lhs:lhs+count]))	
		lhs += count
		tests.append(testlet)

# print(len(tests))

file = open('data.csv', mode='a')
file.write("\"ID\",\"Data\"\n")
for i in range(len(tests)):
	file.write("{},\"{}\"\n".format(i, *tests[i]))
file.close()

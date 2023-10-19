import read
letter="T"

filename="out.csv"
out_data=read.read_data_from(filename,1)

time0=out_data[0]
data=out_data[1]
print(time0)


len0=len(data)

window=30
padding_start=60
padding_end=30

mMin=1000
mMax=0


A=0
for i in data[0:window-1]:
    A+=abs(i)
t=A/window
result=[t for i in range(window)]
for i in range(window,len0-1):
    if A>mMax:
        mMax=A
    if A<mMin:
        mMin=A

    result.append(A/window)
    A=A+(abs(data[i+1])-abs(data[i+1-window]))
result.append(A/window)




import matplotlib.pyplot as plt


plt.plot(time0,result)
plt.show()

standard=(0.97*mMin+0.03*mMax)/window

flag=0

slicing=[]
for i in range(window,len0):

    if result[i]<standard:
        if flag==1:
            slicing.append([start-padding_start,i+padding_end])
            flag=0
        result[i]=0
        continue
    if flag==0:
        start=i
        flag=1

'''
print(len(slicing))

test=[0.0 for i in range(len0)]
for i in slicing:
    test[i[0]:i[1]]=data[i[0]:i[1]]


plt.plot(time0,result)
plt.plot(time0,test)
plt.show()

out=[time0,test]
read.data2CsvFile("final_data/"+letter+".csv",out)


out=[[],[]]

for i in slicing:
    out[0].append(i[0])
    out[1].append(i[1])
read.data2CsvFile("final_data/"+letter+"_index.csv",out)

'''
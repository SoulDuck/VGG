import json
import os
import copy
f=open('forjson.csv')
lines=f.readlines()
new_lines=[]

for line in lines:
    if line.split(',')[1:][0]=='':
        continue;
    new_lines.append(line.split(',')[1:])
    print line.split(',')[1:]
global_dict={}
for i,line in enumerate(new_lines[1:]):
    tmp_dict = {}
    if i%2 ==0: # L

        ext='.png'
        tmp_dict['patientId']=line[6]
        tmp_dict['date'] = line[7]
        tmp_dict['birth'] = line[8]
        tmp_dict['name'] = line[9]
        left_={}
        left_['image']=line[0]+'.png' #
        left_['blendImage']=line[0]+'_blend_ori.png'

        left_['cataract']=line[1] #line[1]
        left_['glaucoma']=line[2] #line[3]
        left_['retina']=line[3]
        left_['artifact']=line[5]
        tmp_dict['left'] = left_
        global_dict=tmp_dict

    else: #R
        tmp_dict=copy.deepcopy(global_dict)


        right_={}

        right_['image']=line[0]+'.png' #
        right_['blendImage']=line[0]+'_blend_ori.png'
        right_['cataract']=line[1]

        right_['glaucoma']=line[2]
        right_['retina']=line[3]
        right_['artifact']=line[5]


        tmp_dict['right'] = right_
        f = open(os.path.join('./tmp_json', tmp_dict['patientId'] + '.txt'), 'w')

        json.dump(tmp_dict , f)
        print tmp_dict
        f.close()



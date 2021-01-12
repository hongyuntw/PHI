#來源 & 你想存哪
sourse_filename = './dataset/development_1.txt'
des_filename = './dataset/development_1.json'

import json
def txt2json(src,des):
  result = []
  flag = 0
  with open(src,'r') as f:
    data = {
        'article':'',
        'id':0,
        'item':[],
    } 
    for line in f:
      line = line.replace('\n','')
      if line=='':
        continue
      if line == '--------------------':
        flag = 0
        result.append(data)
        data['id'] = data['item'][0][0]
        data = {
          'article':'',
          'id':0,
          'item':[],
        }
        continue
      if flag == 0:
        data['article']= line
        flag = 1
      elif flag ==1:
        flag = 2
      elif flag ==2:
        row = line.split('	')
        data['item'].append([int(row[0]),int(row[1]),int(row[2]),row[3],row[4]])
  with open(des,'w', encoding='utf8') as f:
    json.dump(result,f,ensure_ascii=False)
  return

# 執行
txt2json(sourse_filename,des_filename)

# read
with open(des_filename,'r') as f:
  f = json.load(f)
  print(f[0])
  print(f[0]['id'])
  print(f[0]['article'])
  print(f[0]['item'])

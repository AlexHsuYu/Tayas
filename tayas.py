import pandas as pd, numpy as np,os
from . import PROJECT_ROOT
def read_data1(path):
  df1 = pd.read_csv(path,encoding='gbk', sep='\t', names = range(1))
  df1.columns = range(len(df1.columns))
  return df1.loc[0].values, df1.loc[1].values, df1.loc[2].values

def generate_index():
  path = os.path.join(PROJECT_ROOT,'datasets/0826B')
  path_1 =os.path.join(PROJECT_ROOT,'datasets/0826C')
  label_1 = []
  label_2_1 = []
  label_2_2 = []
  label_2_3 = []
  path1 = []
  path2 = []
  for root, dirs, files in os.walk(path):
    for file in files:
      label1 =file.split('_',2)
      if label1[0] == 'sensor' and label1[1]== '1':
        path1.append(os.path.join(root,file))
      if label1[0] == 'sensor' and label1[1]== '2':
        path2.append(os.path.join(root,file))
      if label1[0] == 'label' and label1[1]== '1':
        label_1.append(read_data(os.path.join(root,file)))
      if label1[0] == 'label' and label1[1]== '2':
        label_2_1.append(read_data1(os.path.join(root,file))[0])
        label_2_2.append(read_data1(os.path.join(root,file))[1])
        label_2_3.append(read_data1(os.path.join(root,file))[2])
      
  for root, dirs, files in os.walk(path_1):
    for file in files:
      label1 =file.split('_',2)
      if label1[0] == 'sensor' and label1[1]== '1':
        path1.append(os.path.join(root,file))
      if label1[0] == 'sensor' and label1[1]== '2':
        path2.append(os.path.join(root,file))
      if label1[0] == 'label' and label1[1]== '1':
        label_1.append(read_data(os.path.join(root,file)))
      if label1[0] == 'label' and label1[1]== '2':
        label_2_1.append(read_data1(os.path.join(root,file))[0])
        label_2_2.append(read_data1(os.path.join(root,file))[1])
        label_2_3.append(read_data1(os.path.join(root,file))[2])      
  
  full_df = pd.DataFrame(np.column_stack((path1,path2,label_1,label_2_1,label_2_2,label_2_3)), columns=['path1', 'path2','label_1','label_2_1','label_2_2','label_2_3'])
  full_df.to_csv(os.path.join(PROJECT_ROOT,'datasets/index.csv'), index = False)

def generate_index1():
  df = pd.read_csv(os.path.join(PROJECT_ROOT,'index.csv'))

  df['path1'] = df['path1'].str.replace("./datasets/",os.path.join(PROJECT_ROOT,'./datasets/0826B/'))
  df['path2'] = df['path2'].str.replace("/projectB/0826B/sensor_2/",os.path.join(PROJECT_ROOT,'./datasets/0826C/'))
  df.to_csv(os.path.join(PROJECT_ROOT,'datasets/index.csv'), index = False)
  

def generate_output(): 
  nn_test = nn_test.sort_values(by=['file']).reset_index(drop=True)
  rnn_test = rnn_test.sort_values(by=['file1']).reset_index(drop=True)

  test_result = pd.concat([nn_test, rnn_test], axis = 1)
  filename = test_result['file']
  test_result = test_result.drop(['file','file1'], axis = 1)
  test_result.insert(0,u'試題編號',filename)
  test_result[u"試題編號"] = test_result[u"試題編號"].str.replace("sensor","projectB")
  test_result[u"試題編號"] = test_result[u"試題編號"].str.replace("_1_","_TEST_")    
  test_result.to_csv(os.path.join(PROJECT_ROOT,'datasets/test_result.csv'), index = False)
  test_result = pd.read_csv(os.path.join(PROJECT_ROOT,'datasets/test_result.csv'), names = range(5))
  test_result.columns = ['', 'label1','label2','','']  

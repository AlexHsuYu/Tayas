import pandas as pd, numpy as np, os, keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from .config import Config
from . import index_path, PROJECT_ROOT, test_path, rnn_test_path
from keras.utils import np_utils
from .model import NN, RNN

def NN_train():
    dfs = pd.read_csv(index_path)
    train_df, valid_df = train_test_split(
        dfs,
        test_size= Config.test_size,
        stratify=dfs['label_1'],
        shuffle=True
        )
    train_df =  pd.DataFrame(dfs)    
    full_df = pd.DataFrame()
    val_df = pd.DataFrame()    
    for k, col in train_df.iterrows():        
        try:
            df = pd.read_csv(str(train_df.loc[k,'path1']),names = range(8))  #取 8 ,不取csv的 index
            col2 = []
            train_value = []
            for i in range(2400):                            #取12W 每100  中的最大值
                col2.append(np.mean(df.iloc[i*50:(i+1)*50,1]))
            if np.max(col2) > 40:
                df3 = df.iloc[Config.value_start:Config.value_end,:]
                df3 = preprocessing.scale(df3)
                df3 = pd.DataFrame(df3)
                df3['label'] = np.full(df3.shape[0], col['label_1'])
                full_df = full_df.append(df3,ignore_index=True)            
            else:
                pass
        except FileNotFoundError:
            pass
        
    full_df = full_df.dropna(axis=0)
    X_train = full_df.iloc[:,0:8]
    y_train = full_df.iloc[:,8]    
    y_train = np_utils.to_categorical(y_train)
    nn = NN()
    nn.fit(X_train,y_train,epochs=Config.NN_epochs, batch_size=Config.NN_batch_size,verbose=2)
    nn.save(os.path.join(PROJECT_ROOT,'weights/model.h5'))
    
def NN_test():
    model = keras.models.load_model(os.path.join(PROJECT_ROOT,'weights/model.h5'))
    test_index = pd.DataFrame({})
    test_file = []
    test_category = []
    for root, dirs, files in os.walk(test_path):
        for file in files:
            df = pd.read_csv(os.path.join(root,file))
            col2 = []
            test_value = []
            counts = 0 
            for i in range(2400):                            #取12W 每100  中的最大值
                col2.append(np.mean(df.iloc[i*50:(i+1)*50,2]))
            if np.max(col2) > 40:
                df5 = df.iloc[Config.value_start:Config.value_end,1:9]
                df5 = preprocessing.scale(df5)
                df5 = pd.DataFrame(df5)            
                X_test = df5.iloc[:,0:8]
                pred = model.predict_classes(X_test)
                counts = np.bincount(pred)   
                test_file.append(file.split('.')[0])
                test_category.append(np.argmax(counts))          
            else:
                test_file.append(file.split('.')[0])
                test_category.append(0)

    test_file = pd.DataFrame(test_file, columns = ['file'])
    test_category = pd.DataFrame(test_category,columns =  ['class'])
    test_index  =  pd.concat([test_file,test_category],axis = 1,names=['file', 'class'],sort=True)
    test_index = test_index.sort_values(by=['file'])

    return test_index     

def RNN_train():
    dfs = pd.read_csv(index_path)
    train_df = pd.DataFrame(dfs)

    full_df = pd.DataFrame()
    full_df_V2 = pd.DataFrame()
    for k, col in train_df.iterrows():        
        try:
            df = pd.read_csv(str(train_df.loc[k,'path2']),names = range(8))
            df = df.rolling(window=Config.sliding_window).mean()
            df = df.loc[Config.RNN_value_start:Config.RNN_value_end,:].reset_index(drop=True)
            Standard = StandardScaler()
            df = Standard.fit_transform(df)
            Standard = Normalizer()
            df = Standard.fit_transform(df)
            df = pd.DataFrame(df)
            df['Ra'] = np.full(df.shape[0], col['label_2_1'])
            df['Rz'] = np.full(df.shape[0], col['label_2_2'])
            df['Rmax'] = np.full(df.shape[0], col['label_2_3'])
            full_df = full_df.append(df,ignore_index=True)
        except FileNotFoundError:
            pass

    full_df = full_df.dropna(axis=0)    
    X_train = full_df.iloc[:,0:8]    
    y_train = full_df.iloc[:,8:]
    X_train = np.array(X_train).reshape(X_train.shape[0], 1, 8)
    rnn = RNN()
    rnn.fit(X_train, y_train, batch_size=Config.RNN_batch_size,epochs=Config.RNN_epochs, verbose=2)
    rnn.save(os.path.join(PROJECT_ROOT,'weights/model_V2.h5'))

def RNN_test():
    model = keras.models.load_model(os.path.join(PROJECT_ROOT,'weights/model_V2.h5'))
    test_index = pd.DataFrame({})
    test_file = []
    test_Ra = []
    test_Rz = []
    test_Rmax = []
    for root, dirs, files in os.walk(rnn_test_path):
        for file in files:
            df = pd.read_csv(os.path.join(root,file),names = range(8))
            df = df.rolling(window=Config.sliding_window).mean()
            df = df.loc[Config.RNN_value_start:Config.RNN_value_end,:].reset_index(drop=True)
            Standard = StandardScaler()
            df = Standard.fit_transform(df)
            Standard = Normalizer() 
            df = Standard.fit_transform(df)
            df = pd.DataFrame(df)
            valid_value=[]
            X_test = df.loc[:,0:8]
            X_test = np.array(X_test).reshape(X_test.shape[0], 1, 8)
            pred = model.predict(X_test)
            test_file.append(file.split('.')[0])
            test_Ra.append(pred[:,0].sum()/df.shape[0])
            test_Rz.append(pred[:,1].sum()/df.shape[0])
            test_Rmax.append(pred[:,2].sum()/df.shape[0])
    test_file = pd.DataFrame(test_file, columns = ['file1'])
    test_Ra = pd.DataFrame(test_Ra,columns =  ['Ra'])
    test_Rz = pd.DataFrame(test_Rz,columns =  ['Rz'])
    test_Rmax = pd.DataFrame(test_Rmax,columns =  ['Rmax'])
    test_index  =  pd.concat([test_file,test_Ra,test_Rz,test_Rmax],axis = 1,names=['file1', 'Ra','Rz','Rmax'],sort=True) 
    test_index = test_index.sort_values(by=['file1'])
    return test_index 

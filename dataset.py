import pandas as pd
from sklearn.preprocessing import StandardScaler
def get_MNIST():
    train_set = pd.read_csv('./data/mnist_train.csv')
    val_set = pd.read_csv('./data/mnist_test.csv')
    train_x = train_set.iloc[:, 1:].to_numpy()
    train_y = train_set['label'].to_numpy()
    val_x = val_set.iloc[:, 1:].to_numpy()
    val_y = val_set['label'].to_numpy()
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    val_x=scaler.transform(val_x)
    print("Shape of MNIST data is: ",train_x.shape,train_y.shape,val_x.shape,val_y.shape)
    return train_x[:],train_y[:],val_x,val_y

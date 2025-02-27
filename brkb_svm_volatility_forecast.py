import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
import plotly.graph_objects as go

# Step 1: Define neural network model
class Net(nn.Module):
    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, 50)
        self.fc2 = nn.Linear(50, out_size)
        self.act = nn.ReLU()

    def forward(self, X):
        X = self.fc1(X)
        X = self.act(X)
        X = self.fc2(X)
        return X

# Step 2: Create normalization utility
class Scale:
    def __init__(self, dim):
        self.dim = dim

    def normalize(self, X):
        self.mins = []
        self.maxs = []
        arr = np.array(X)
        for i in range(self.dim):
            self.mins.append(np.min(arr[:, i]))
            self.maxs.append(np.max(arr[:, i]))
        return (arr - self.mins) / (np.array(self.maxs) - self.mins)

    def denormalize(self, X):
        arr = np.array(X)
        return arr * (np.array(self.maxs) - self.mins) + self.mins

# Step 3: Split data into train/test sets
def TrainTestSplit(ratio):
    def wrapper(func):
        def split(*args, **kwargs):
            X, Y = func(*args, **kwargs)
            split_idx = int(ratio * len(X))
            trX, teX = X[:split_idx], X[split_idx:]
            trY = Y[:split_idx]
            return trX, trY, teX
        return split
    return wrapper

# Step 4: Calculate volatility
def CalcVol(prices, lookback=100, horizon=30):
    returns = prices[1:] / prices[:-1] - 1.0
    vol = [np.std(returns[i-lookback:i]) for i in range(lookback, len(returns))]
    return vol

# Step 5: Prepare volatility sequence dataset
@TrainTestSplit(0.7)
def SeqData(vol, seq_len=30, pred_len=10):
    X, Y = [], []
    for i in range(seq_len, len(vol) - pred_len + 1):
        X.append(vol[i-seq_len:i])
        Y.append(vol[i:i+pred_len])
    return np.array(X), np.array(Y)

# Step 6: Prepare good/bad volatility dataset
@TrainTestSplit(0.7)
def ClassifyVol(prices, lookback=100, horizon=30):
    ret = lambda seq: np.prod([1 + r for r in seq]) - 1.0
    X, Y = [], []
    returns = prices[1:] / prices[:-1] - 1.0
    vol = CalcVol(prices)
    ret_seq = returns[lookback:]
    for i in range(lookback, len(vol) - horizon + 1):
        X.append(vol[i-lookback:i])
        Y.append(1 if ret(ret_seq[i:i+horizon]) > 0 else 0)
    return np.array(X), np.array(Y)

# Step 7: Load and preprocess data
df = pd.read_csv('BRKB.csv')[::-1]
df = df[df['date'] >= '2023-05-01']
prices = df['adjClose'].values
vol = CalcVol(prices)

# Step 8: Generate datasets
seqX, seqY, seq_teX = SeqData(vol)
clsX, clsY, cls_teX = ClassifyVol(prices)

# Step 9: Normalize data
scale_seqX = Scale(30)
scale_seqY = Scale(10)
scale_teX = Scale(30)
scale_clsX = Scale(100)
scale_cls_teX = Scale(100)

X = scale_seqX.normalize(seqX)
Y = scale_seqY.normalize(seqY)
teX = scale_teX.normalize(seq_teX)
cls_trX = scale_clsX.normalize(clsX)
cls_teX = scale_cls_teX.normalize(cls_teX)

# Step 10: Train SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(cls_trX, clsY)
pred_cls = svm.predict(cls_teX)
prob_cls = svm.predict_proba(cls_teX)

pos_vol = sum(pred_cls)
neg_vol = len(pred_cls) - pos_vol
accuracy = np.mean(prob_cls[:, 0])

# Step 11: Prepare tensors for neural network
XX = torch.stack([torch.tensor(x, dtype=torch.float32) for x in X])
YY = torch.stack([torch.tensor(y, dtype=torch.float32) for y in Y])
teXX = torch.stack([torch.tensor(x, dtype=torch.float32) for x in teX])

# Step 12: Train neural network
model = Net(30, 10)
loss_fn = nn.MSELoss()
opt = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(400):
    pred = model(XX)
    loss = loss_fn(pred, YY)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"Epoch: {epoch + 1}")

# Step 13: Generate forecast
with torch.no_grad():
    forecast_tensor = model(teXX)
forecast = forecast_tensor.numpy()
forecast = scale_seqY.denormalize(forecast)[-1]

# Step 14: Visualize results with Plotly
hist_vol = vol[-30:]
x_hist = list(range(len(hist_vol)))
x_pred = list(range(len(hist_vol), len(hist_vol) + len(forecast)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_hist, y=hist_vol, mode='lines', name='Historical Volatility',
                         line=dict(color='#FF6B6B', width=2)))
fig.add_trace(go.Scatter(x=x_pred, y=forecast, mode='lines', name='Forecasted Volatility',
                         line=dict(color='#4ECDC4', width=2, dash='dash')))

fig.update_layout(
    title=f'Good Volatility: {pos_vol} | Bad Volatility: {neg_vol} | Accuracy: {accuracy:.2f}',
    xaxis_title='Time',
    yaxis_title='Volatility',
    template='plotly_dark',
    plot_bgcolor='rgb(40, 40, 40)',
    paper_bgcolor='rgb(40, 40, 40)',
    font=dict(color='white'),
    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=0.5),
    margin=dict(l=50, r=50, t=50, b=50),
    showlegend=True
)

fig.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy

df_dev = pd.read_csv('treatment_response_development.csv')

print(df_dev.describe())

print(df_dev.columns)

print(df_dev.Age)

sum_column_2 = df_dev['Weight_loss_percent'].sum()

print("Sum of values in column 2:", sum_column_2)

print(df_dev.head())

print(df_dev.info())

df_dev.drop(columns=['Smoking'], inplace=True)   #selects a column named smoking and drops it and it is removed from df_dev
df_dev.dropna(inplace=True)                      #This line removes any rows with missing values from the DataFrame df_dev

print(df_dev.head())                       # when this is executed it shows 15 cols instead of 16

df_dev = df_dev.join(pd.get_dummies(df_dev['Sex']))
df_dev['F'] = df_dev['F'].astype(int)
df_dev['M'] = df_dev['M'].astype(int)

print(df_dev.head()) # check whether dummy cols are converted to integer and added in data frame

def diff_grade_num(row):
    if row['Differentiation_grade'] == 'G1':
        val = 1
    elif row['Differentiation_grade'] == 'G2':
        val = 2
    elif row['Differentiation_grade'] == 'G3':
        val = 3
    else:
        val = 0
    return val

def overall_stage_num(row):
    if row['Overall_stage'] == 'I':
        val = 1
    elif row['Overall_stage'] == 'II':
        val = 2
    elif row['Overall_stage'] == 'III':
        val = 3
    else:
        val = 4
    return val


df_dev['Differentiation_grade_num'] = df_dev.apply(diff_grade_num, axis=1)
df_dev['Overall_stage_num'] = df_dev.apply(overall_stage_num, axis=1)

print(df_dev.head())

df_input = df_dev[['Age', 'F', 'M', 'Differentiation_grade_num', 'Survival_time_days', 
                     'Overall_stage_num', 'Weight_loss_percent']]
df_output = df_dev['Complete_response_probability']

X_train, X_test, y_train, y_test = train_test_split(df_input, df_output, 
                                                    test_size=0.33, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# modeling with leniar regression

reg = LinearRegression().fit(np.array(X_train), y_train) # model fitting
y_test_predictions = reg.predict(np.array(X_test)) # getting the outcomes

# results presentation: regression plot and performance scores
sns.regplot(x = y_test, y = y_test_predictions)
plt.xlabel('p_true')
plt.ylabel('p_predicted')
plt.title('Linear regression results')
plt.show()

print('hiiii')
print ('R2: {}, MSE: {}'.format(r2_score(y_test, y_test_predictions), 
                                mean_squared_error(y_test, y_test_predictions)))

# modeling with neural net

X_train_nnet = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train_nnet = torch.tensor(np.array(y_train), dtype=torch.float32).reshape(-1, 1)
X_test_nnet = torch.tensor(np.array(X_test), dtype=torch.float32)
y_test_nnet = torch.tensor(np.array(y_test), dtype=torch.float32).reshape(-1, 1)

# defininng the model  sets up neural network

nnet = nn.Sequential(
    nn.Linear(7, 24),
    nn.ReLU(),
    nn.Linear(24, 12),
    nn.ReLU(),
    nn.Linear(12, 6),
    nn.ReLU(),
    nn.Linear(6, 1)
)

loss_fn = nn.MSELoss()  
optimizer = optim.Adam(nnet.parameters(), lr=0.0001)

n_epochs = 100   
batch_size = 10 
batch_start = torch.arange(0, len(X_train), batch_size)

best_mse = np.inf   
best_weights = None
history = []

# training process
for epoch in range(n_epochs):
    nnet.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train_nnet[start:start+batch_size]
            y_batch = y_train_nnet[start:start+batch_size]
            # forward pass
            y_pred = nnet(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    nnet.eval()
    y_pred = nnet(X_test_nnet)
    mse = loss_fn(y_pred, y_test_nnet)
    mse = float(mse)
    history.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(nnet.state_dict())
 
nnet.load_state_dict(best_weights)

print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()

#regression plot + performance scores

y_test_predictions_nn = nnet(X_test_nnet)

sns.regplot(x = y_test, y = y_test_predictions_nn.detach().numpy().flatten())
plt.xlabel('p_true')
plt.ylabel('p_predicted')
plt.title('Neural net results')
plt.show()

print ('R2: {}, MSE: {}'.format(r2_score(y_test, y_test_predictions_nn.detach().numpy().flatten()), 
                                mean_squared_error(y_test, y_test_predictions_nn.detach().numpy().flatten())))  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred1=lr.predict(X_test)

accuracy_score(y_test,y_pred1)

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

y_pred2=knn.predict(X_test)

accuracy_score(y_test,y_pred2)

rf= RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred3=rf.predict(X_test)

accuracy_score(y_test,y_pred3)

sgd = SGDClassifier()

for i in range(len(X_train)):
    sgd.partial_fit(X_train[i:i+1],y_train[i:i+1],classes=['R','M'])

score=sgd.score(X_test,y_test)

print("Acc:",score)

final_data = pd.DataFrame({'Models':['LR','KNN','RF','SGD'],
             'ACC':[accuracy_score(y_test,y_pred1),
                   accuracy_score(y_test,y_pred2),
                   accuracy_score(y_test,y_pred3),
                   score]})

joblib.dump(knn1,'rock_mine_prediction_model')

def predict_data():

    file_path = filedialog.askopenfilename(title="Select prediction data file", filetypes=(("CSV files", "*.csv"),))

    data = pd.read_csv(file_path,header=None)    
    
    knn_model = joblib.load('rock_mine_prediction_model')

    predictions = knn_model.predict(data)

    print(predictions)
    if predictions == 'M':
        s = "Mine"
    else:
        s= "Rock"
    messagebox.showinfo(title="Predictions", message=str(s))
    
window = tk.Tk()

window.geometry("300x100")

window.configure(bg="light gray")

screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

x = (screen_width - window.winfo_reqwidth()) / 2
y = (screen_height - window.winfo_reqheight()) / 2

window.geometry("+%d+%d" % (x, y))

button = tk.Button(window, text="Select prediction data", command=predict_data)
button.pack()

button_width = 150
button_height = 30
button_x = (300 - button_width) / 2
button_y = (100 - button_height) / 2

button.place(relx=button_x/300, rely=button_y/100, relwidth=button_width/300, relheight=button_height/100)

window.mainloop()

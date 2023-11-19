from flask import Flask, render_template, request

app = Flask(__name__)

# Initial product list
products = ["Potatoes", "Brocolli", "Brinjal","Watermelon","Muskmelon","Pineapple"]
# Maximum number of products to display
max_display = 3 #frames
display = []
algo=""
hits=0
total=0
faults=0
fifoindex=0
history=[]
histindex=0
showalgos=1
dummy=history
@app.route('/',methods=['GET', 'POST'])
def index():
    global algo,max_display,hits,total,faults
    showalgos=1
    if request.method == 'POST':
        algo = request.form.get('algo')
        hits=0
        total=0
        faults=0
    return render_template('index.html', products=products, display=display, algo=algo,max_display=max_display,showalgos=showalgos)

@app.route('/replace', methods=['POST'])
def replace():
    global algo,max_display,hits,total,faults,fifoindex, history, histindex
    selected_product = request.form.get('product')
    history.append(selected_product)
    f=0
    print(display)
    for item in display: 
        if selected_product == item["name"]:
            item["lfucount"]+=1
            item["lrucount"]=0
            f=1
            hits+=1
        else:
            item["lrucount"]+=1
    if f==0 and len(display)<max_display: 
        faults+=1
        display.append({
            "name": selected_product,
            "lfucount": 0,
            "lrucount": 0,
        })
        print("appended")  
    elif f==0 and len(display)==max_display:
        faults+=1
        if algo == 'FIFO':
            print("fifo")
            display[fifoindex]={
                'name':selected_product,
                'lfucount':0,
                'lrucount':0
            }
            fifoindex=(fifoindex+1)%max_display
        elif algo == 'Optimal':
            print("optimal")
            import numpy as np
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import LabelEncoder  
            import random
            # def generate_next_reference_optimal(page, pageindex):
            #     future_references = page[pageindex + 1:]
            #     unique_future_pages = np.unique(future_references)
            #     if len(unique_future_pages) == 0:
            #         return np.random.choice(page)
            #     optimal_page = max(unique_future_pages, key=lambda x: np.argmax(future_references == x))
            #     return optimal_page
            # def train_linear_regression_model(page):
            #     next_references_optimal = [generate_next_reference_optimal(history, i) for i in range(len(history) - 1)]
            #     next_references_optimal.append(np.random.choice(history))
            #     X_train = np.array(history[:-1]).reshape(-1, 1)
            #     label_encoder = LabelEncoder()
            #     y_train = label_encoder.fit_transform(next_references_optimal[:-1])
            #     X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            #     model = LinearRegression()
            #     model.fit(X_train, y_train)
            #     return model
            def predict_last_reference(model, current_reference):
                current_reference = np.array(current_reference).reshape(1, -1)
                predicted_reference = model.predict(current_reference)
                return predicted_reference[0]
            random_item = random.choice(display)["name"]
            for item in display: 
                if random_item == item["name"]:
                    item["name"]=selected_product
                    item["lfucount"]+=1
                    item["lrucount"]=0
    
            #linear_regression_model = train_linear_regression_model(dummy)
            current_reference_to_predict = selected_product
            #predicted_last_reference = predict_last_reference(linear_regression_model, current_reference_to_predict)
            #print(predicted_last_reference)
            
        elif algo == 'LRU':
            print("lru")
            maxlrucount=display[0]["lrucount"]
            indexcount=0
            maxlruindex=0
            for item in display: 
                if item["lrucount"] > maxlrucount:
                    maxlrucount=item["lrucount"]
                    maxlruindex=indexcount
                indexcount+=1
            print(maxlruindex)
            display[maxlruindex]={
                "name":selected_product,
                "lrucount":0,
                "lfucount":0,
            }
        elif algo == 'LFU':
            print("lfu")
            minlfucount=display[0]["lfucount"]
            indexcount=0
            minlfuindex=0
            for item in display: 
                if item["lfucount"] < minlfucount:
                    minlfucount=item["lfucount"]
                    minlfuindex=indexcount
                indexcount+=1
            print(minlfuindex)
            display[minlfuindex]={
                "name":selected_product,
                "lrucount":0,
                "lfucount":0,
            }
    showalgos=0
    return render_template('index.html', products=products, display=display,algo=algo,max_display=max_display,showalgos=showalgos,hits=hits,faults=faults)

if __name__ == '__main__':
    app.run(debug=True)

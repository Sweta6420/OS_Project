from flask import Flask, render_template, request

app = Flask(__name__)

# Initial product list
products = ["prod1", "prod2", "prod3"]
# Maximum number of products to display
max_display = 2
# FIFO queue to keep track of displayed products
display_queue = []

@app.route('/')
def index():
    return render_template('index.html', products=products, display_queue=display_queue)

@app.route('/replace', methods=['POST'])
def replace():
    if len(display_queue) < max_display:
        # Display queue is not full, add the selected product
        selected_product = request.form['product']
        display_queue.append(selected_product)
    else:
        # Display queue is full, use FIFO to replace the oldest product
        replaced_product = display_queue.pop(0)
        selected_product = request.form['product']
        display_queue.append(selected_product)

    return render_template('index.html', products=products, display_queue=display_queue)

if __name__ == '__main__':
    app.run(debug=True)

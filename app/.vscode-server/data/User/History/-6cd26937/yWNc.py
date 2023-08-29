# Import packages
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib

app = dash.Dash(__name__)

# Define styles for different elements
headline_style = {"fontSize": 24, "textAlign": "center", "color": "#FCF3CF"}
instruction_style = {"fontSize": 16, "textAlign": "center", "color": "#FEF9E7", "margin": "10px"}
input_style = {"width": "150px", "margin": "10px", "color": "#000000", "backgroundColor": "#FFFFFF"}
submit_button_style = {"textAlign": "center", "marginTop": "20px", "backgroundColor": "#F4D03F"}
car_price_style = {"fontSize": 20, "textAlign": "center", "marginTop": "20px", "color": "#FFFFFF"}

app.layout = html.Div(
    style={"backgroundColor": "#0B5345", "padding": "20px"},
    children=[
        html.H1("Car Price Prediction", style=headline_style),
        html.Div(
            "Fill in the values below to predict the car price:",
            style=instruction_style,
        ),
        html.Div(
            [
                html.Label("Max power (bhp):", style={"color": "#FFFFFF"}),
                dcc.Input(id="max_power", type="number", style=input_style),
            ],
            style={"display": "inline-block"},
        ),
        html.Div(
            [
                html.Label("Mileage (kmpl):", style={"color": "#FFFFFF"}),
                dcc.Input(id="mileage", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Div(
            [
                html.Label("Engine size (cc):", style={"color": "#FFFFFF"}),
                dcc.Input(id="engine", type="number", style=input_style),
            ],
            style={"display": "inline-block", "marginLeft": "20px"},
        ),
        html.Div(
            style={"textAlign": "center"},  # Centering the input boxes
            children=[
                html.Button("Submit", id="submit", style=submit_button_style),
                html.Div(id="car_price", style=car_price_style),
            ],
        ),
    ],
)

# Load the trained model
model = joblib.load('Car-price.model')

# Create a callback function to predict the car price and display it on the page
@app.callback(
    Output("car_price", "children"),
    [Input("submit", "n_clicks")],
    [State("max_power", "value"), State("mileage", "value"), State("engine", "value")],
)
def predict_car_price(n_clicks, max_power, mileage, engine_size):
    if n_clicks is None or n_clicks == 0:
        return ""

    input_data = [[max_power, mileage, engine_size]]
    input_data = model['scaler'].transform(input_data)
    car_price = model['model'].predict(input_data)[0]

    return f"The predicted car price is ${car_price:.2f}."

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)




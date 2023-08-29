# Import packages
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import joblib

app = dash.Dash(__name__)

layout = html.Div(
    [
        html.H1("Car Price Prediction"),
        html.Div(
            [
                html.Label("Max power (bhp):"),
                dcc.Input(id="max_power", type="number"),
            ]
        ),
        html.Div(
            [
                html.Label("Mileage (kmpl):"),
                dcc.Input(id="mileage", type="number"),
            ]
        ),
        html.Div(
            [
                html.Label("Engine size (cc):"),
                dcc.Input(id="engine", type="number"),
            ]
        ),
        html.Button("Submit", id="submit"),
        html.Div(id="car_price"),
    ]
)

# Load the trained model
model = joblib.load('Car-Price.model')  # Upload model's file name

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
    app.layout = layout  # Assign the defined layout to the app
    app.run_server(debug=True)



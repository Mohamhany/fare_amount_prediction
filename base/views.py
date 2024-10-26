from django.shortcuts import render
from django.http import JsonResponse
import joblib  
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'catboost_regressor_model.pkl')
model = joblib.load(model_path)
model_scale = os.path.join(os.path.dirname(__file__), '..', 'scaler.pkl')
scaler=joblib.load(model_scale)
def home(request):
    if request.method == 'POST':
        passenger_count = int(request.POST['passenger_count'])
        hour = int(request.POST['hour'])
        day = int(request.POST['day'])
        month = int(request.POST['month'])
        weekday = int(request.POST['weekday'])
        year = int(request.POST['year'])
        distance = float(request.POST['distance'])
        bearing = float(request.POST['bearing'])
        car_condition = request.POST['car_condition']
        weather = request.POST['weather']

        # Map categorical inputs to numeric values
        car_condition_bad = 1 if car_condition == 'Bad' else 0
        car_condition_very_good = 1 if car_condition == 'Very Good' else 0
        weather_sunny = 1 if weather == 'sunny' else 0
        weather_windy = 1 if weather == 'windy' else 0

        
        data = [
            passenger_count, hour, day, month, weekday, year,
            distance, bearing, car_condition_bad,
            car_condition_very_good, weather_sunny, weather_windy
        ]
        scaled_prediction = model.predict(data)

        
        prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1)).flatten()[0]
        
        return render(request, 'home.html',{'prediction': prediction})
    else:
        return render(request, 'home.html')

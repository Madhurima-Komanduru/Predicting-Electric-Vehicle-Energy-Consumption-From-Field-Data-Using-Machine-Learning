import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from .models import EVPrediction, UserRegistrationModel
from django.contrib.auth.models import User

# ===================== TRAINING VIEW =====================
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def training(request):
    base_dir = settings.BASE_DIR
    data_path = os.path.join(base_dir, 'media', 'EV_Energy_Consumption_Dataset.csv')

    df = pd.read_csv(data_path)
    df.dropna(inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df.drop(columns=['timestamp'], inplace=True)

    categorical_cols = ["Driving_Mode", "Road_Type", "Traffic_Condition", "Weather_Condition"]
    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns])
    df = df.select_dtypes(include=['number'])

    X = df.drop('Energy_Consumption_kWh', axis=1)
    y = df['Energy_Consumption_kWh']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LinearRegression": LinearRegression(),
        # "ElasticNet": ElasticNet(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        # "XGBoost": XGBRegressor(),
        # "LightGBM": LGBMRegressor()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        # Save each model
        joblib.dump({"model": model, "features": X_train.columns.tolist()},
                    os.path.join(base_dir, 'media', f'ev_{name.lower()}_model.pkl'))

    # Regression comparison graph
    results_df = pd.DataFrame(results).T
    plt.figure(figsize=(8, 5))
    results_df[['MAE', 'RMSE']].plot(kind='bar')
    plt.title("Model Comparison - MAE & RMSE")
    plt.ylabel("Error")
    plt.tight_layout()
    reg_graph = os.path.join(base_dir, 'media', 'model_comparison.png')
    plt.savefig(reg_graph)
    plt.close()

    # Classification for confusion matrix
    median_value = y.median()
    y_class = (y > median_value).astype(int)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)

    acc = accuracy_score(y_test_c, y_pred_c)
    clf_report = classification_report(y_test_c, y_pred_c, output_dict=False)

    cm = confusion_matrix(y_test_c, y_pred_c)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'High'])
    disp.plot(cmap='Blues')
    plt.title(f"RandomForest Classifier\nAccuracy: {acc:.2f}")
    cm_graph = os.path.join(base_dir, 'media', 'confusion_matrix.png')
    plt.savefig(cm_graph)
    plt.close()

    return render(request, 'users/accuracy.html', {
        'results': results_df.to_html(classes='table table-striped', float_format="%.4f"),
        'graph_url': '/media/model_comparison.png',
        'confusion_url': '/media/confusion_matrix.png',
        'clf_report': clf_report
    })


# ===================== PREDICTION VIEW =====================
def prediction(request):
    prediction_value = None
    error_message = None

    model_path = os.path.join(settings.BASE_DIR, 'media', 'ev_randomforest_model.pkl')
    if not os.path.exists(model_path):
        error_message = "Model not trained yet. Please train the model first."
    else:
        try:
            model_dict = joblib.load(model_path)
            model = model_dict["model"]
            model_features = model_dict["features"]
        except Exception:
            error_message = "Saved model is not in the correct format. Please retrain."

    if request.method == "POST" and not error_message:
        try:
            # --- Get input values ---
            Speed_kmh = float(request.POST.get("Speed_kmh"))
            Acceleration_ms2 = float(request.POST.get("Acceleration_ms2"))
            Battery_State = float(request.POST.get("Battery_State_%"))
            Driving_Mode = request.POST.get("Driving_Mode")
            Road_Type = request.POST.get("Road_Type")
            Traffic_Condition = request.POST.get("Traffic_Condition")
            Weather_Condition = request.POST.get("Weather_Condition")
            Humidity = float(request.POST.get("Humidity_%"))
            Wind_Speed = float(request.POST.get("Wind_Speed_ms"))
            Vehicle_Weight = float(request.POST.get("Vehicle_Weight_kg"))
            Distance_Travelled = float(request.POST.get("Distance_Travelled_km"))

            # --- Prepare DataFrame ---
            input_data = pd.DataFrame([{
                "Speed_kmh": Speed_kmh,
                "Acceleration_ms2": Acceleration_ms2,
                "Battery_State_%": Battery_State,
                "Driving_Mode": Driving_Mode,
                "Road_Type": Road_Type,
                "Traffic_Condition": Traffic_Condition,
                "Weather_Condition": Weather_Condition,
                "Humidity_%": Humidity,
                "Wind_Speed_ms": Wind_Speed,
                "Vehicle_Weight_kg": Vehicle_Weight,
                "Distance_Travelled_km": Distance_Travelled
            }])

            categorical_cols = ["Driving_Mode", "Road_Type", "Traffic_Condition", "Weather_Condition"]
            input_data = pd.get_dummies(input_data, columns=categorical_cols)

            # --- Align with training model columns ---
            for col in model_features:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model_features]

            # --- Make prediction ---
            prediction_value = model.predict(input_data)[0]

            # --- Save prediction to DB using session user ---
            user_obj = None
            user_id = request.session.get('id')
            if user_id:
                from users.models import UserRegistrationModel
                try:
                    user_obj = UserRegistrationModel.objects.get(id=user_id)
                except UserRegistrationModel.DoesNotExist:
                    pass

            EVPrediction.objects.create(
                user=user_obj,
                Speed_kmh=Speed_kmh,
                Acceleration_ms2=Acceleration_ms2,
                Battery_State=Battery_State,
                Driving_Mode=Driving_Mode,
                Road_Type=Road_Type,
                Traffic_Condition=Traffic_Condition,
                Weather_Condition=Weather_Condition,
                Humidity=Humidity,
                Wind_Speed=Wind_Speed,
                Vehicle_Weight=Vehicle_Weight,
                Distance_Travelled=Distance_Travelled,
                Predicted_Energy=prediction_value
            )

        except Exception as e:
            error_message = f"Error in prediction: {str(e)}"

    return render(request, "users/prediction.html", {
        "prediction": prediction_value,
        "error": error_message
    })



# ===================== VIEW DATASET =====================
def ViewDataset(request):
    dataset = os.path.join(settings.MEDIA_ROOT, 'EV_Energy_Consumption_Dataset.csv')
    df = pd.read_csv(dataset, nrows=100)
    df = df.to_html(index=None)
    return render(request, 'users/viewData.html', {'data': df})


# ===================== USER REGISTRATION =====================
def UserRegisterActions(request):
    if request.method == 'POST':
        user = UserRegistrationModel(
            name=request.POST['name'],
            loginid=request.POST['loginid'],
            password=request.POST['password'],
            mobile=request.POST['mobile'],
            email=request.POST['email'],
            locality=request.POST['locality'],
            address=request.POST['address'],
            city=request.POST['city'],
            state=request.POST['state'],
            status='waiting'
        )
        user.save()
        messages.success(request, "Registration successful!")
    return render(request, 'UserRegistrations.html') 


# ===================== USER LOGIN =====================
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if check.status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not Activated')
        except Exception:
            messages.success(request, 'Invalid Login id or password')
    return render(request, 'UserLogin.html', {})


# ===================== HOME & INDEX =====================
def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})

def index(request):
    return render(request, "index.html")


from django.shortcuts import render
from users.models import EVPrediction, UserRegistrationModel

def PreviousResults(request):
    results = []

    # Get logged-in user ID from session
    user_id = request.session.get('id')
    if user_id:
        try:
            user_obj = UserRegistrationModel.objects.get(id=user_id)
            # Fetch all predictions of this user, newest first
            results = EVPrediction.objects.filter(user=user_obj).order_by('-timestamp')
        except UserRegistrationModel.DoesNotExist:
            results = []

    return render(request, 'users/previous_results.html', {'results': results})



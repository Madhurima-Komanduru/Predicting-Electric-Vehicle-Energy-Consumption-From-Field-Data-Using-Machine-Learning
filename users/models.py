from django.db import models

# User Registration Model
class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=10)
    email = models.EmailField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    address = models.TextField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100, default='waiting')

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'user_registrations'


# EV Prediction Model
class EVPrediction(models.Model):
    user = models.ForeignKey(UserRegistrationModel, on_delete=models.CASCADE, null=True, blank=True)
    
    Speed_kmh = models.FloatField()
    Acceleration_ms2 = models.FloatField()
    Battery_State = models.FloatField()
    Driving_Mode = models.CharField(max_length=50)
    Road_Type = models.CharField(max_length=50)
    Traffic_Condition = models.CharField(max_length=50)
    Weather_Condition = models.CharField(max_length=50)
    Humidity = models.FloatField()
    Wind_Speed = models.FloatField()
    Vehicle_Weight = models.FloatField()
    Distance_Travelled = models.FloatField()
    Predicted_Energy = models.FloatField()

    # Automatically store the timestamp when a record is created
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.Predicted_Energy:.2f} kWh"

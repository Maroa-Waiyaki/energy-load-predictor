from django.urls import path
from .views import predict_energy

urlpatterns = [
    path('', predict_energy, name='predict_energy'),
]

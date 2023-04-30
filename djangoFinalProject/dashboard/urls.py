from django.urls import path
from . import views

urlpatterns = [
    path('report/mushrooms', views.mushrooms_report_page, name='mushrooms_report'),
]

from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("process/", views.process_article, name="process_article"),
]

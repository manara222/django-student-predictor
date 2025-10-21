from django.contrib import admin
from django.urls import path, include  # <-- 1. أضيفي include هنا

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictor_app.urls')), # <-- 2. أضيفي هذا السطر بالكامل
]
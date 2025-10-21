from django.urls import path
from . import views

urlpatterns = [
    # هذا السطر يعني: عندما يطلب المستخدم الصفحة الرئيسية للموقع
    # قومي بتشغيل دالة 'home' الموجودة في ملف views.py
    path('', views.home, name='home'),
]
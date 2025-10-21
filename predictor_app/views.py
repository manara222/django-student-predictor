from django.shortcuts import render
import joblib
import pandas as pd

# =====================================================================
# الصفحة الرئيسية التي ستحتوي على الفورم
# =====================================================================
def home(request):
    # إذا كان المستخدم قد أرسل بيانات (ضغط على زر التنبؤ)
    if request.method == 'POST':
        try:
            # 1. استقبال البيانات من الفورم الذي سنبنيه
            #    نستخدم .get() لتجنب الأخطاء إذا كانت القيمة فارغة
            hours_studied = float(request.POST.get('hours_studied'))
            previous_scores = float(request.POST.get('previous_scores'))
            sleep_hours = float(request.POST.get('sleep_hours'))
            attendance = float(request.POST.get('attendance'))
            extracurricular = request.POST.get('extracurricular')
            internet_access = request.POST.get('internet_access')
            school_type = request.POST.get('school_type')
            gender = request.POST.get('gender')

            # 2. تحميل "عقل" المشروع (الـ Pipeline المحفوظ)
            model_pipeline = joblib.load('ml_model/grade_predictor_pipeline.pkl')

            # 3. تجهيز البيانات للتنبؤ
            #    يجب أن نضع البيانات في DataFrame بنفس أسماء الأعمدة وترتيبها الذي تدرب عليه النموذج
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Previous_Scores': [previous_scores],
                'Sleep_Hours': [sleep_hours],
                'Attendance': [attendance],
                'Extracurricular_Activities': [extracurricular],
                'Internet_Access': [internet_access],
                'School_Type': [school_type],
                'Gender': [gender]
            })

            # 4. استخدام الـ Pipeline للتنبؤ بالدرجة
            prediction = model_pipeline.predict(input_data)[0]

            # تقريب النتيجة لأقرب رقمين عشريين
            predicted_score = round(prediction, 2)
            
            # 5. إرسال النتيجة إلى صفحة جديدة لعرضها
            return render(request, 'result.html', {'prediction': predicted_score})

        except Exception as e:
            # في حالة حدوث أي خطأ، سنعرض رسالة للمستخدم
            error_message = f"حدث خطأ ما. تأكد من إدخال جميع القيم بشكل صحيح. الخطأ: {e}"
            return render(request, 'home.html', {'error': error_message})

    # إذا كانت هذه هي المرة الأولى التي يفتح فيها المستخدم الصفحة، سيعرض الفورم فقط
    return render(request, 'home.html')
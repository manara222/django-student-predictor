# =============================================================================
# 1. استيراد المكتبات الأساسية
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

print("Step 1: المكتبات تم استيرادها بنجاح.")

# =============================================================================
# 2. تحميل البيانات
# =============================================================================
file_path = 'ml_model/StudentPerformanceFactors.csv'
try:
    df = pd.read_csv(file_path)
    print("Step 2: البيانات تم تحميلها بنجاح.")
except FileNotFoundError:
    print(f"خطأ: لم يتم العثور على الملف في المسار التالي: {file_path}")
    exit()

# =============================================================================
# 3. تجهيز البيانات (باستخدام أسماء الأعمدة الصحيحة)
# =============================================================================
# تحديد المتغير الهدف (الدرجة التي نريد التنبؤ بها)
target = 'Exam_Score'

# تحديد المتغيرات المساعدة التي سنستخدمها
# قمنا باختيار مجموعة متنوعة من المتغيرات الرقمية والنصية من قائمتك
features = [
    'Hours_Studied', 'Previous_Scores', 'Sleep_Hours', 'Attendance',
    'Extracurricular_Activities', 'Internet_Access', 'School_Type', 'Gender'
]

X = df[features]
y = df[target]

# فصل الأعمدة الرقمية عن الفئوية (النصية) تلقائيًا
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nStep 3: تجهيز البيانات...")
print(f"المتغيرات الرقمية المختارة: {numerical_features}")
print(f"المتغيرات الفئوية المختارة: {categorical_features}")
print(f"المتغير الهدف: {target}")


# =============================================================================
# 4. بناء الـ Pipeline للمعالجة والتدريب
# =============================================================================
# إنشاء معالج للمتغيرات الرقمية (توحيد القياس)
numeric_transformer = StandardScaler()

# إنشاء معالج للمتغيرات الفئوية (تحويلها لأرقام)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# دمج المعالجين في خطوة معالجة واحدة
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# اختيار النموذج
model = RandomForestRegressor(n_estimators=100, random_state=42)

# بناء الـ Pipeline الكامل الذي يربط المعالجة بالنموذج
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

print("Step 4: الـ Pipeline تم بناؤه بنجاح للتعامل مع كل أنواع البيانات.")

# =============================================================================
# 5. تقسيم البيانات، تدريب النموذج، وتقييمه
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nStep 5: جاري تدريب النموذج...")
full_pipeline.fit(X_train, y_train)
print("... النموذج تم تدريبه بنجاح!")

# التنبؤ باستخدام بيانات الاختبار
y_pred = full_pipeline.predict(X_test)

# تقييم أداء النموذج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- نتائج تقييم النموذج ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
print("--------------------------")

# =============================================================================
# 6. حفظ الـ Pipeline المدرب
# =============================================================================
output_path = 'ml_model/grade_predictor_pipeline.pkl'
joblib.dump(full_pipeline, output_path)

print(f"\nStep 6: النموذج تم حفظه بنجاح في المسار التالي: {output_path}")
print("🎉 تهانينا يا منار! أنتِ الآن جاهزة للمرحلة التالية وهي بناء تطبيق الويب.")
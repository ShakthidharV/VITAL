# ehr/urls.py
from rest_framework import routers
from django.urls import path, include
from .views import PatientViewSet, PractitionerViewSet, ObservationViewSet

router = routers.DefaultRouter()
router.register(r"observations", ObservationViewSet, basename="observation")

# ehr/urls.py
from django.urls import path, include
from . import views
from rest_framework import routers


# If you already created router earlier, keep it. Example for API routers omitted for brevity.

app_name = "patient"
urlpatterns = [
    path("", views.home, name="home"),
    path("api/", include(router.urls)),
    path("patient/register/", views.RegisterView.as_view(), name="register"),
    path("patient/login/", views.LoginView.as_view(), name="login"),
    path("patient/logout/", views.LogoutView.as_view(), name="logout"),
    path("patient/dashboard/", views.DashboardView.as_view(), name="dashboard"),
    path("patient/profile/", views.ProfileUpdateView.as_view(), name="profile"),
    path("patient/records/", views.RecordListView.as_view(), name="records"),
    path("patient/records/<uuid:pk>/", views.RecordDetailView.as_view(), name="record_detail"),

    path("doctor/login/", views.DoctorLoginView.as_view(), name="doctor_login"),
    path("doctor/dashboard/", views.doctor_dashboard, name="doctor_dashboard"),
    path("doctor/patient/<uuid:patient_id>/", views.doctor_patient_detail, name="patient_detail"),
    path("doctor/patient/<uuid:patient_id>/observation/add/", views.observation_add, name="observation_add"),
    path("doctor/patient/<uuid:patient_id>/observation/<uuid:obs_id>/edit/", views.observation_edit, name="observation_edit"),
    path("patient/self/ml-entry/", views.patient_entry, name="patient_entry_self"),
    path("patient/self/ml-submit/", views.patient_submit, name="patient_submit_self"),
]



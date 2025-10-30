
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.decorators.http import require_http_methods
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.urls import reverse_lazy
from django.http import HttpResponseForbidden
from django.conf import settings
from django.http import HttpResponseBadRequest
from django.utils import timezone

from .forms import PatientRegisterForm, PatientProfileForm, CustomAuthenticationForm, ObservationForm
from .models import Practitioner, Patient, Observation
from .decorators import practitioner_required
from . import models

# ehr/views.py — append these imports near the top if not present
import json
import hashlib

from rest_framework import viewsets, permissions, authentication
from .models import Observation
from .serializers import DeidentifiedObservationSerializer


from .ml_nhanes_module import list_models, get_expected_features, predict_risk


# Create your views here.
# ehr/views.py
from rest_framework import viewsets
from .models import Patient, Practitioner, Observation
from .serializers import PatientSerializer, PractitionerSerializer, ObservationSerializer

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer

class PractitionerViewSet(viewsets.ModelViewSet):
    queryset = Practitioner.objects.all()
    serializer_class = PractitionerSerializer

class ObservationViewSet(viewsets.ModelViewSet):
    queryset = Observation.objects.all()
    serializer_class = ObservationSerializer



class ObservationViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only API for deidentified observations.
    Auth: TokenAuthentication. Permissions: IsAuthenticated.
    """
    queryset = Observation.objects.all()
    serializer_class = DeidentifiedObservationSerializer
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        qs = super().get_queryset()
        # optional filters by code and date-range (YYYY-MM-DD or full ISO)
        code = self.request.query_params.get("code")
        if code:
            qs = qs.filter(code=code)
        start = self.request.query_params.get("start")
        if start:
            qs = qs.filter(effective_date__gte=start)
        end = self.request.query_params.get("end")
        if end:
            qs = qs.filter(effective_date__lte=end)
        return qs


# ehr/views.py
from django.shortcuts import render

def home(request):
    # Optional: you can add context entries for dynamic counts/notifications later
    return render(request, "home.html", {})


class RegisterView(View):
    def get(self, request):
        form = PatientRegisterForm()
        return render(request, "patient/register.html", {"form": form})
    def post(self, request):
        form = PatientRegisterForm(request.POST)
        if form.is_valid():
            patient = form.save()
            login(request, patient.user)
            return redirect("patient:dashboard")
        return render(request, "patient/register.html", {"form": form})

class LoginView(View):
    def get(self, request):
        form = CustomAuthenticationForm()
        return render(request, "patient/login.html", {"form": form})
    def post(self, request):
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("patient:dashboard")
        return render(request, "patient/login.html", {"form": form})

class LogoutView(View):
    def post(self, request):
        logout(request)
        return redirect("patient:login")

@method_decorator(login_required, name="dispatch")
class DashboardView(View):
    def get(self, request):
        # find patient linked to user
        patient = getattr(request.user, "patient", None)
        if not patient:
            # Option: create a patient record stub if needed; here show message
            return render(request, "patient/dashboard.html", {"error": "No patient profile linked to your account."})
        recent_obs = Observation.objects.filter(patient=patient).order_by("-effective_date")[:8]
        return render(request, "patient/dashboard.html", {"patient": patient, "recent_obs": recent_obs})

# ehr/views.py snippet for ProfileUpdateView
@method_decorator(login_required, name="dispatch")
class ProfileUpdateView(View):
    def get(self, request):
        patient = request.user.patient
        form = PatientProfileForm(instance=patient)
        return render(request, "patient/profile_form.html", {"form": form, "patient": patient})

    def post(self, request):
        patient = request.user.patient
        form = PatientProfileForm(request.POST, instance=patient)
        if form.is_valid():
            form.save()
            return redirect("patient:dashboard")
        return render(request, "patient/profile_form.html", {"form": form, "patient": patient})


@method_decorator(login_required, name="dispatch")
class RecordListView(View):
    def get(self, request):
        patient = request.user.patient
        records = Observation.objects.filter(patient=patient).order_by("-effective_date")
        return render(request, "patient/records_list.html", {"patient": patient, "records": records})



@method_decorator(login_required, name="dispatch")
class RecordDetailView(View):
    """
    Patient-facing record detail view that:
      - allows a patient to view only their own Observation records
      - allows a practitioner (or staff/superuser) to view any Observation by pk
    """
    def get(self, request, pk):
        # 1) If the current user is a patient -> enforce ownership
        patient = getattr(request.user, "patient", None)
        if patient is not None:
            # patient viewing own record (strict)
            record = get_object_or_404(Observation, pk=pk, patient=patient)
            return render(request, "patient/record_detail.html", {"patient": patient, "record": record})

        # 2) If not a patient, allow practitioner/staff/superuser to view by pk
        if hasattr(request.user, "practitioner") or request.user.is_staff or request.user.is_superuser:
            # find the record (if it doesn't exist, 404 is appropriate)
            record = get_object_or_404(Observation, pk=pk)
            patient = record.patient
            return render(request, "patient/record_detail.html", {"patient": patient, "record": record})

        # 3) all other authenticated users are denied
        return HttpResponseForbidden("You are not authorized to view this record.")


class DoctorLoginView(View):
    template_name = "doctor/login.html"

    def get(self, request):
        form = CustomAuthenticationForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("patient:doctor_dashboard")   # adjust if you namespaced the URL
        return render(request, self.template_name, {"form": form})

@practitioner_required
def doctor_dashboard(request):
    # list patients (optionally restrict to those in practitioner's care)
    q = request.GET.get("q")
    patients = Patient.objects.all().order_by("family")
    if q:
        patients = patients.filter(models.Q(given__icontains=q) | models.Q(family__icontains=q) | models.Q(identifier__icontains=q))
    return render(request, "doctor/dashboard.html", {"practitioner": request.user.practitioner, "patients": patients})

@practitioner_required
def doctor_patient_detail(request, patient_id):
    patient = get_object_or_404(Patient, pk=patient_id)
    records = Observation.objects.filter(patient=patient).order_by("-effective_date")
    return render(request, "doctor/patient_detail.html", {"patient": patient, "records": records, "practitioner": request.user.practitioner})

@practitioner_required
def observation_add(request, patient_id):
    patient = get_object_or_404(Patient, pk=patient_id)
    if request.method == "POST":
        form = ObservationForm(request.POST)
        if form.is_valid():
            obs = form.save(commit=False)
            obs.patient = patient
            # set performer to practitioner name if you want; ensure model field accepts Practitioner or string
            # If Observation.performer is a FK to Practitioner change accordingly:
            try:
                obs.performer = request.user.practitioner
            except Exception:
                obs.performer = None
            obs.save()
            return redirect("patient:patient_detail", patient_id=patient.id)
    else:
        form = ObservationForm()
    return render(request, "doctor/observation_form.html", {"form": form, "patient": patient})

@practitioner_required
def observation_edit(request, patient_id, obs_id):
    patient = get_object_or_404(Patient, pk=patient_id)
    obs = get_object_or_404(Observation, pk=obs_id, patient=patient)
    if request.method == "POST":
        form = ObservationForm(request.POST, instance=obs)
        if form.is_valid():
            form.save()
            return redirect("patient:patient_detail", patient_id=patient.id)
    else:
        form = ObservationForm(instance=obs)
    return render(request, "doctor/observation_form.html", {"form": form, "patient": patient, "edit": True})


# Add these patient-facing views (paste anywhere in the file, e.g., after DashboardView)
def _compute_deid_hash(identifier: str) -> str:
    salt = getattr(settings, "ML_PATIENT_HASH_SALT", "change_this_in_prod")
    return hashlib.sha256(f"{identifier}{salt}".encode("utf-8")).hexdigest()

def _coerce_feature_values(expected_features, raw_dict):
    out = {}
    for f in expected_features:
        v = raw_dict.get(f)
        if v is None or v == "":
            out[f] = float("nan")   # let imputer handle missing
            continue
        try:
            out[f] = float(v)
        except Exception:
            # keep categoricals as strings
            out[f] = str(v)
    return out

def patient_entry(request):
    """
    Public patient self-check page.
    If the user has a linked Patient record, it will be used; otherwise submission is saved as anonymous (hashed id).
    """
    models = list_models()
    schema = {m: get_expected_features(m) for m in models}
    patient_obj = getattr(request.user, "patient", None) if request.user.is_authenticated else None
    return render(request, "patient/patient_entry.html", {
        "models": models,
        "schema_json": json.dumps(schema),
        "patient": patient_obj,
    })

@require_http_methods(["POST"])
def patient_submit(request):
    """
    Handles patient-submitted features. Accepts form fields named exactly as features.
    Saves an Observation and renders result page.
    """
    disease = request.POST.get("disease")
    if not disease:
        return HttpResponseBadRequest("Missing disease")

    expected = get_expected_features(disease)
    if expected is None:
        return HttpResponseBadRequest("Unknown disease")

    # Prefer a single JSON 'features' field; else read individual fields from POST
    features_raw = request.POST.get("features")
    if features_raw:
        try:
            raw = json.loads(features_raw)
        except Exception:
            return HttpResponseBadRequest("Invalid features JSON")
    else:
        # build raw dict from POST — pick only expected keys
        raw = {k: v for k, v in request.POST.items()}

    raw_subset = {f: raw.get(f) for f in expected}
    features = _coerce_feature_values(expected, raw_subset)

    # resolve patient / deid hash
    patient_obj = None
    if request.user.is_authenticated:
        patient_obj = getattr(request.user, "patient", None)
        if patient_obj:
            deid = _compute_deid_hash(str(patient_obj.pk))
        else:
            deid = _compute_deid_hash(str(request.user.pk))
    else:
        anon_id = request.POST.get("anon_id") or request.POST.get("email") or request.META.get("REMOTE_ADDR") or "anon"
        deid = _compute_deid_hash(str(anon_id))

    # call model
    try:
        prob = float(predict_risk(disease, features))
    except Exception as e:
        return HttpResponseBadRequest(f"Prediction error: {e}")

    # threshold from settings
    thresholds = getattr(settings, "ML_RISK_THRESHOLDS", {})
    threshold = float(thresholds.get(disease, 0.2))
    alert_flag = prob >= threshold

    # Save as Observation (fill required fields)
    obs = Observation.objects.create(
        patient=patient_obj,
        code=disease,
        value=str(round(prob, 6)),
        unit="probability",
        effective_date=timezone.now(),
        performer=None,
        remarks="ML self-check",
        disease_key=disease,
        risk_score=prob,
        features=features,
        deidentified_patient_hash=deid,
        alert=alert_flag
    )

    return render(request, "patient/patient_result.html", {
        "disease": disease,
        "risk": prob,
        "threshold": threshold,
        "alert": alert_flag,
        "observation": obs
    })

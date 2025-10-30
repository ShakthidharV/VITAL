from django.db import models

# Create your models here.
# ehr/models.py
import uuid
from django.db import models
from django.contrib.auth.models import User

from .utils import deidentify_patient



class Patient(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="patient")
    given = models.CharField(max_length=200)        # first name(s)
    family = models.CharField(max_length=200)       # last name
    birth_date = models.DateField(null=True, blank=True)
    gender = models.CharField(max_length=16, null=True, blank=True)
    identifier = models.CharField(max_length=200, null=True, blank=True)  # MRN / external id
    phone = models.CharField(max_length=50, null=True, blank=True)
    address = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"{self.family}, {self.given}"
# ehr/models.py (add this; if Practitioner already exists adapt it)

class Practitioner(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.OneToOneField(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="practitioner")
    name = models.CharField(max_length=300)
    identifier = models.CharField(max_length=200, null=True, blank=True)  # e.g., DOC-00001
    specialty = models.CharField(max_length=200, null=True, blank=True)
    phone = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.name or (self.user.get_full_name() if self.user else "Practitioner")


# ehr/models.py — replace the existing Observation model with this (or add fields to it)
class Observation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    patient = models.ForeignKey(Patient, related_name="observations", on_delete=models.CASCADE, null=True, blank=True)
    deidentified_patient_hash = models.CharField(max_length=128, null=True, blank=True)
    code = models.CharField(max_length=200)
    value = models.CharField(max_length=200)
    unit = models.CharField(max_length=50, null=True, blank=True)
    effective_date = models.DateTimeField()
    performer = models.ForeignKey(Practitioner, null=True, blank=True, on_delete=models.SET_NULL)
    remarks = models.TextField(null=True, blank=True, help_text="Doctor's free-text remarks / clinical notes")

    # --- new ML-related fields ---
    disease_key = models.CharField(max_length=128, null=True, blank=True)
    risk_score = models.FloatField(null=True, blank=True)
    features = models.JSONField(null=True, blank=True)  # stores the submitted features
    deidentified_patient_hash = models.CharField(max_length=128, null=True, blank=True)
    alert = models.BooleanField(default=False)

    class Meta:
        ordering = ["-effective_date"]

    def __str__(self):
        return f"{self.code}={self.value}{(' '+self.unit) if self.unit else ''}"
    
    def save(self, *args, **kwargs):
        # compute hash if patient set and hash empty or patient changed
        if self.patient:
            try:
                self.deidentified_patient_hash = deidentify_patient(self.patient)
            except Exception:
                # fail-safe: don't prevent save if deid fails
                pass
        super().save(*args, **kwargs)


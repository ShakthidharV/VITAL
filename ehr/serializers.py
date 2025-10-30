# ehr/serializers.py
from rest_framework import serializers
from .models import Patient, Practitioner, Observation

class PatientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Patient
        fields = "__all__"

class PractitionerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Practitioner
        fields = "__all__"

class ObservationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Observation
        fields = "__all__"

# Deidentified serializer: only expose safe fields
class DeidentifiedObservationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Observation
        fields = [
            "id",
            "deidentified_patient_hash",
            "code",
            "value",
            "unit",
            "effective_date",
            "disease_key",
            "risk_score",
            "features",
            "alert",
        ]
        read_only_fields = fields

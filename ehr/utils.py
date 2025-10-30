import hmac, hashlib
from django.conf import settings

def deidentify_patient(patient):
    # deterministic HMAC-SHA256 over patient UUID (or identifier) using secret salt
    secret = settings.DEID_SALT.encode()
    # prefer patient.identifier if available else patient.id
    base = (patient.identifier or str(patient.id)).encode()
    digest = hmac.new(secret, base, hashlib.sha256).hexdigest()
    return digest  # store this in Observation.deidentified_patient_hash

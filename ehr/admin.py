from django.contrib import admin

# Register your models here.
# ehr/admin.py
from django.contrib import admin
from .models import Patient, Practitioner, Observation

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ("family", "given", "identifier", "birth_date", "gender")
    search_fields = ("family", "given", "identifier")


@admin.register(Observation)
class ObservationAdmin(admin.ModelAdmin):
    list_display = ("code","value","patient","effective_date", "performer")
    search_fields = ("code","value","patient__family","remarks")
    list_filter = ("code",)
    readonly_fields = ()


@admin.register(Practitioner)
class PractitionerAdmin(admin.ModelAdmin):
    list_display = ("name", "identifier", "specialty", "user")
    search_fields = ("name", "identifier", "user__username", "user__email")

# ehr/forms.py
from django import forms
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.models import User
from .models import Patient, Observation


class PatientRegisterForm(forms.ModelForm):
    email = forms.EmailField(required=True)
    password = forms.CharField(widget=forms.PasswordInput, min_length=8)

    class Meta:
        model = Patient
        fields = ("given", "family", "birth_date", "gender", "phone", "address")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for fname in self.fields:
            if fname == "password":
                continue
            self.fields[fname].widget.attrs.update({"class": "form-control"})
        self.fields["email"].widget.attrs.update({"class": "form-control", "placeholder": "you@example.com"})
        self.fields["password"].widget.attrs.update({"class": "form-control"})

    def save(self, commit=True):
        data = self.cleaned_data
        # Create Django user first (user.id will be available after creation)
        username = (data.get("email") or f"patient_{data.get('given')}_{data.get('family')}").lower()
        user = User.objects.create_user(
            username=username,
            email=data.get("email"),
            password=data.get("password"),
            first_name=data.get("given"),
            last_name=data.get("family"),
        )

        # create patient instance but do not commit yet
        patient = super().save(commit=False)
        patient.user = user

        # set a readable Patient ID derived from the new user.id
        # format: PAT-000001, PAT-000002, etc.
        # user.id is an integer auto-assigned by Django once saved
        patient.identifier = f"PAT-{user.id:06d}"

        if commit:
            patient.save()
        return patient


class PatientProfileForm(forms.ModelForm):
    class Meta:
        model = Patient
        fields = ("given", "family", "birth_date", "gender", "phone", "address")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for f in self.fields.values():
            f.widget.attrs.update({"class": "form-control"})

class CustomAuthenticationForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # style username & password fields
        self.fields["username"].widget.attrs.update({"class": "form-control", "placeholder": "Email or username"})
        self.fields["password"].widget.attrs.update({"class": "form-control"})



class ObservationForm(forms.ModelForm):
    class Meta:
        model = Observation
        fields = ("code", "value", "unit", "effective_date", "remarks")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # make text inputs bootstrap, remarks uses textarea
        for name, field in self.fields.items():
            if name == "remarks":
                field.widget = forms.Textarea(attrs={"class":"form-control", "rows":"5", "placeholder":"Enter clinical remarks, findings or notes..."})
            else:
                field.widget.attrs.update({"class":"form-control"})

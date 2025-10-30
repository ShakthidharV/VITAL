# your_app/context_processors.py
def role_flags(request):
    is_practitioner = False
    if request.user.is_authenticated:
        try:
            # import inside function to avoid circular import on startup
            from .models import Practitioner
            is_practitioner = Practitioner.objects.filter(user=request.user).exists()
        except Exception:
            is_practitioner = False
    return {
        "is_practitioner": is_practitioner
    }

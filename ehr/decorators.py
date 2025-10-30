# ehr/decorators.py
from django.shortcuts import redirect
from functools import wraps
from django.http import HttpResponseForbidden

def practitioner_required(view_func):
    @wraps(view_func)
    def _wrapped(request, *args, **kwargs):
        # allow superusers/staff as well if desired
        if not request.user.is_authenticated:
            return redirect("patient:login")   # reuse patient login or create doctor login route
        if hasattr(request.user, "practitioner") and request.user.practitioner is not None:
            return view_func(request, *args, **kwargs)
        # optionally allow staff:
        if request.user.is_staff or request.user.is_superuser:
            return view_func(request, *args, **kwargs)
        return HttpResponseForbidden("You are not authorized to access this page.")
    return _wrapped

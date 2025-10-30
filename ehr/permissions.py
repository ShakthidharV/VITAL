from rest_framework.permissions import BasePermission

class IsResearcher(BasePermission):
    """
    Allow access only to users with is_staff or group 'researcher' or JWT claim 'researcher': True.
    Adjust logic to your auth model.
    """
    def has_permission(self, request, view):
        user = request.user
        if not user or not user.is_authenticated:
            return False
        # 1) staff/admin allowed
        if user.is_staff:
            return True
        # 2) group-based
        if user.groups.filter(name="researcher").exists():
            return True
        # 3) JWT custom claim (if using SimpleJWT with custom claims)
        if getattr(request, "auth", None):
            # request.auth is validated token; check claim if present
            try:
                if request.auth.get("researcher", False):
                    return True
            except Exception:
                pass
        return False

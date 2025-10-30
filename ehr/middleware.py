import logging
logger = logging.getLogger("ehr.audit")

class AuditMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        # only log researcher API observation reads
        if request.path.startswith("/api/observations") and request.user.is_authenticated:
            logger.info(
                "RESEARCH_ACCESS user=%s path=%s method=%s status=%s",
                request.user.username,
                request.path,
                request.method,
                response.status_code,
            )
        return response

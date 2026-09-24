from fastapi import HTTPException, status


class NexusException(HTTPException):
    """Base exception for NEXUS platform errors."""
    def __init__(self, status_code: int, detail: str, code: str = "NEXUS_ERROR"):
        super().__init__(status_code=status_code, detail={"message": detail, "code": code})


class AuthenticationFailedException(NexusException):
    def __init__(self, detail: str = "Incorrect email or password"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            code="AUTH_FAILED"
        )


class EntityNotFoundException(NexusException):
    def __init__(self, entity_name: str, entity_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{entity_name} with id '{entity_id}' was not found.",
            code="NOT_FOUND"
        )


class PermissionDeniedException(NexusException):
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            code="FORBIDDEN"
        )


class ValidationException(NexusException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            code="VALIDATION_ERROR"
        )

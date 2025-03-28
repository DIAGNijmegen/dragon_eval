# Adapted from evalutils version 0.4.2
class EvalUtilsError(Exception):
    pass


class FileLoaderError(EvalUtilsError):
    pass


class ValidationError(EvalUtilsError):
    pass


class ConfigurationError(EvalUtilsError):
    pass

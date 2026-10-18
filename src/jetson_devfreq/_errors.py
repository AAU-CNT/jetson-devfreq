"""Exception hierarchy for jetson_devfreq."""


class DevfreqError(Exception):
    """Base exception for all devfreq errors."""


class DevfreqPermissionError(DevfreqError):
    """Raised when a sysfs/debugfs access fails due to insufficient privileges."""


class DevfreqNotFoundError(DevfreqError):
    """Raised when an expected sysfs path does not exist on this platform."""


class DevfreqGovernorError(DevfreqError):
    """Raised when a requested governor is not available."""


class DevfreqFrequencyError(DevfreqError):
    """Raised when a requested frequency is not available or did not settle."""

class InvalidWktPointArgument(Exception):
    """Raised when the WKT string is not a valid point"""

    pass


class InvalidDateRangeArgument(Exception):
    """Raised when the daterange string is not valid"""

    pass


class SearchException(Exception):
    """Raised when search endpoint returned a non 200 statuscode"""

    pass

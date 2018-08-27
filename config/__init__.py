from __future__ import absolute_import

# Create this object before importing the following imports, since they edit the list
option_list = {}


from . import (  # noqa
    log_file,
)


def config_value(option):
    """
    Return the current configuration value for the given option
    """
    return option_list[option]
